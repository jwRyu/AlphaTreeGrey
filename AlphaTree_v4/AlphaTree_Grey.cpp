#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
using namespace std;

#define OUTPUT_FNAME "C:/Users/jwryu/RUG/2018/AlphaTree/test.dat"

#define INPUTIMAGE_DIR	"C:/Users/jwryu/Google Drive/RUG/2018/AlphaTree/imgdata/Grey"
#define INPUTIMAGE_DIR_COLOUR	"C:/Users/jwryu/Google Drive/RUG/2018/AlphaTree/imgdata/Colour" //colour images are used after rgb2grey conversion
#define REPEAT 1
#define RUN_TSE_ONLY 0
#define RUN_MAX_ONLY 1

#define DEBUG 0
#define max(a,b) (a)>(b)?(a):(b)
#define min(a,b) (a)>(b)?(b):(a)

#define DELAYED_ANODE_ALLOC		1
#define HQUEUE_COST_AMORTIZE	1

#define NULL_LEVELROOT		0xffffffff
#define ANODE_CANDIDATE		0xfffffffe

#define dimg_idx_v(pidx) ((pidx)<<1)
#define dimg_idx_h(pidx) ((pidx)<<1)+1

#define LEFT_AVAIL(pidx,width)			(((pidx) % (width)) != 0)
#define RIGHT_AVAIL(pidx,width)			(((pidx) % (width)) != ((width) - 1))
#define UP_AVAIL(pidx,width)				((pidx) > ((width) - 1))
#define DOWN_AVAIL(pidx,width,imgsz)		((pidx) < (imgsz) - (width))

#define A		1.10184
#define SIGMA	-2.6912
#define B		-0.0608
#define M		0.03

//Memory allocation reallocation schemes
#define TSE 0
#define MAXIMUM 1
#define LINEAR 2
#define EXP 3
int mem_scheme = -1;
double size_init[4] = { -1, 1, .2, .15 };
double size_mul[4] = { 1, 1, 1, 2 };
double size_add[4] = { .05, 0, 0.15, 0 };

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned long uint32;
typedef unsigned long long uint64;
typedef char int8;
typedef short int16;
typedef long int32;
typedef long long int64;

typedef uint8 pixel; //designed for 8-bit images

double nrmsd;

size_t memuse, max_memuse;

#if DEBUG
void* buf;
uint64 bufsize;
void save_buf(void* src, uint64 size)
{
	memcpy(buf, src, size);
	bufsize = size;
}
uint8 isChanged(void *src)
{
	uint64 i;
	for (i = 0; i < bufsize; i++)
	{
		if (((uint8*)buf)[i] != ((uint8*)src)[i])
			return 1;
	}
	return 0;
}
#endif

inline void* Malloc(size_t size)
{
	void* pNew = malloc(size + sizeof(size_t));

	memuse += size;
	max_memuse = max(memuse, max_memuse);

	*((size_t*)pNew) = size;
	return (void*)((size_t*)pNew + 1);
}

inline void* Realloc(void* ptr, size_t size)
{
	void* pOld = (void*)((size_t*)ptr - 1);
	size_t oldsize = *((size_t*)pOld);
	void* pNew = realloc(pOld, size + sizeof(size_t));

	if (pOld != pNew)
		max_memuse = max(memuse + size, max_memuse);
	else
		max_memuse = max(memuse + size - oldsize, max_memuse);
	memuse += size - oldsize;

	*((size_t*)pNew) = size;
	return (void*)((size_t*)pNew + 1);
}

inline void Free(void* ptr)
{
	size_t size = *((size_t*)ptr - 1);
	memuse -= size;
	free((void*)((size_t*)ptr - 1));
}

typedef struct HQueue
{
	int32 *queue, *bottom, *cur;
	int64 qsize;
	int32 min_level, max_level;
}HQueue;


HQueue* hqueue_new(uint64 qsize, int32 *dhist, int32 dhistsize)
{
	int32 i;
	HQueue* hqueue = (HQueue*)Malloc(sizeof(HQueue));
	hqueue->queue = (int32*)Malloc((size_t)qsize * sizeof(int32));
	hqueue->bottom = (int32*)Malloc((size_t)(dhistsize + 1) * sizeof(int32));
	hqueue->cur = (int32*)Malloc((size_t)(dhistsize + 1) * sizeof(int32));

	hqueue->qsize = qsize;
	hqueue->min_level = hqueue->max_level = dhistsize;

	int sum_hist = 0;
	for (i = 0; i < dhistsize; i++)
	{
		hqueue->bottom[i] = hqueue->cur[i] = sum_hist;
		sum_hist += dhist[i];
	}
	hqueue->bottom[dhistsize] = 0;
	hqueue->cur[dhistsize] = 1;

	return hqueue;
}

void hqueue_free(HQueue* hqueue)
{
	Free(hqueue->queue);
	Free(hqueue->bottom);
	Free(hqueue->cur);
	Free(hqueue);
}

inline void hqueue_push(HQueue* hqueue, int32 newidx, int32 level)
{
	hqueue->min_level = min(level, hqueue->min_level);
#if DEBUG
	assert(level < hqueue->max_level);
	assert(hqueue->cur[level] < hqueue->qsize);
#endif
	hqueue->queue[hqueue->cur[level]++] = newidx;
}

inline int32 hqueue_pop(HQueue* hqueue)
{
	return hqueue->queue[--hqueue->cur[hqueue->min_level]];
}

inline void hqueue_find_min_level(HQueue* hqueue)
{
	while (hqueue->bottom[hqueue->min_level] == hqueue->cur[hqueue->min_level])
		hqueue->min_level++;
}

typedef struct AlphaNode
{
	int32 area;
	uint8 level;  /* alpha of flat zone */
	double sumPix;
	pixel minPix;
	pixel maxPix;
	int32 parentidx;
} AlphaNode;

typedef struct AlphaTree
{
	int32 maxSize;
	int32 curSize;
	int32 height, width, channel;
	AlphaNode* node;
	int32* parentAry;
} AlphaTree;


#if DELAYED_ANODE_ALLOC
inline int32 NewAlphaNode(AlphaTree* tree)
{
	if (tree->curSize == tree->maxSize)
	{
		printf("Reallocating...\n");
		tree->maxSize = min(2 * tree->height * tree->width, (int32)(size_mul[mem_scheme] * tree->maxSize) + (int32)(2 * tree->height * tree->width * size_add[mem_scheme]));

		tree->node = (AlphaNode*)Realloc(tree->node, tree->maxSize * sizeof(AlphaNode));
	}
	return tree->curSize++;
}
#else
inline int32 NewAlphaNode(AlphaTree* tree, uint8 level)
{
	AlphaNode *pNew = tree->node + tree->curSize;

	if (tree->curSize == tree->maxSize)
	{
		printf("Reallocating...\n");
		tree->maxSize = min(2 * tree->height * tree->width, (int32)(size_mul[mem_scheme] * tree->maxSize) + (int32)(2 * tree->height * tree->width * size_add[mem_scheme]));

		tree->node = (AlphaNode*)Realloc(tree->node, tree->maxSize * sizeof(AlphaNode));
		pNew = tree->node + tree->curSize;
	}
	pNew->level = level;
	pNew->minPix = (uint8)-1;
	pNew->maxPix = 0;
	pNew->sumPix = 0.0;
	pNew->parentidx = 0;
	pNew->area = 0;

	return tree->curSize++;
}
#endif

#if DELAYED_ANODE_ALLOC
inline void connectPix2Node(AlphaTree *tree, int32* parentAry, int32 pidx, pixel pix_val, int32 *levelroot, int32 level)
{
	AlphaNode* pNode;
	int32 iNode = levelroot[level];
	if (iNode == ANODE_CANDIDATE)
	{
		iNode = NewAlphaNode(tree);
		levelroot[level] = iNode;
		parentAry[pidx] = iNode;
		pNode = tree->node + iNode;
		pNode->area = 1;
		pNode->level = level;
		pNode->maxPix = pix_val;
		pNode->minPix = pix_val;
		pNode->sumPix = (double)pix_val;
	}
	else
	{
		pNode = tree->node + iNode;
		parentAry[pidx] = iNode;
		pNode->area++;
		pNode->maxPix = max(pNode->maxPix, pix_val);
		pNode->minPix = min(pNode->minPix, pix_val);
		pNode->sumPix += pix_val;
	}
}
#else
inline void connectPix2Node(int32* parentAry, int32 pidx, pixel pix_val, AlphaNode* pNode, int32 iNode)
{
	parentAry[pidx] = iNode;
	pNode->area++;
	pNode->maxPix = max(pNode->maxPix, pix_val);
	pNode->minPix = min(pNode->minPix, pix_val);
	pNode->sumPix += pix_val;
}
#endif

#if DELAYED_ANODE_ALLOC
inline void connectNode2Node(AlphaTree *tree, int32* levelroot, AlphaNode* pChild, int32 level)
{
	AlphaNode *pPar;
	int32 iPar = levelroot[level];
	if (iPar == ANODE_CANDIDATE)
	{
		iPar = NewAlphaNode(tree);
		levelroot[level] = iPar;
		pPar = tree->node + iPar;
		pChild->parentidx = iPar;
		pPar->area = pChild->area;
		pPar->maxPix = pChild->maxPix;
		pPar->minPix = pChild->minPix;
		pPar->sumPix = pChild->sumPix;
		pPar->level = level;
	}
	else
	{
		pPar = tree->node + iPar;
		pChild->parentidx = iPar;
		pPar->area += pChild->area;
		pPar->maxPix = max(pChild->maxPix, pPar->maxPix);
		pPar->minPix = min(pChild->minPix, pPar->minPix);
		pPar->sumPix += pChild->sumPix;
	}
}
#else
inline void connectNode2Node(AlphaNode* pPar, int32 iPar, AlphaNode* pNode)
{
	pNode->parentidx = iPar;
	pPar->area += pNode->area;
	pPar->maxPix = max(pNode->maxPix, pPar->maxPix);
	pPar->minPix = min(pNode->minPix, pPar->minPix);
	pPar->sumPix += pNode->sumPix;
}
#endif

void compute_dimg(uint8* dimg, int32* dhist, pixel* img, int32 height, int32 width, int32 channel)
{
	int32 dimgidx, imgidx, stride_w = width, i, j;

	imgidx = dimgidx = 0;
	for (i = 0; i < height - 1; i++)
	{
		for (j = 0; j < width - 1; j++)
		{
			dimg[dimgidx] = (uint8)(abs((int)img[imgidx + stride_w] - (int)img[imgidx]));
			dhist[dimg[dimgidx++]]++;
			dimg[dimgidx] = (uint8)(abs((int)img[imgidx + 1] - (int)img[imgidx]));
			dhist[dimg[dimgidx++]]++;
			imgidx++;
		}
		dimg[dimgidx] = (uint8)(abs((int)img[imgidx + stride_w] - (int)img[imgidx]));
		dhist[dimg[dimgidx++]]++;
		dimgidx++;
		imgidx++;
	}
	for (j = 0; j < width - 1; j++)
	{
		dimgidx++;
		dimg[dimgidx] = (uint8)(abs((int)img[imgidx + 1] - (int)img[imgidx]));
		dhist[dimg[dimgidx++]]++;
		imgidx++;
	}
}

inline uint8 is_visited(uint8* isVisited, int32 p)
{
	return (isVisited[p>>3] >> (p & 7)) & 1;
}

inline void visit(uint8* isVisited, int32 p)
{
	isVisited[p >> 3] = isVisited[p >> 3] | (1 << (p & 7));
}

void Flood(AlphaTree* tree, pixel* img, int32 height, int32 width, int32 channel)
{
	int32 imgsize, dimgsize, nredges, max_level, current_level, next_level, x0, p, dissim;
	int32 numlevels;
	HQueue* hqueue;
	int32 *dhist;
	uint8 *dimg;
	int32 iChild, *levelroot;
	uint8 *isVisited;
	int32 *pParentAry;
	
	imgsize = width * height;
	nredges = width * (height - 1) + (width - 1) * height;
	dimgsize = 2 * width * height; //To make indexing easier
	numlevels = 1 << (8 * sizeof(uint8));

	dhist = (int32*)Malloc((size_t)numlevels * sizeof(int32));
	dimg = (uint8*)Malloc((size_t)dimgsize * sizeof(uint8));
	levelroot = (int32*)Malloc((int32)(numlevels + 1) * sizeof(int32));
	isVisited = (uint8*)Malloc((size_t)((imgsize + 7) >> 3));
	for (p = 0; p < numlevels; p++)
		levelroot[p] = NULL_LEVELROOT;
	memset(dhist, 0, (size_t)numlevels * sizeof(int32));
	memset(isVisited, 0, (size_t)((imgsize + 7) >> 3));

	max_level = (uint8)(numlevels - 1);
	
	compute_dimg(dimg, dhist, img, height, width, channel);
	dhist[max_level]++;
	hqueue = hqueue_new(nredges + 1, dhist, numlevels);
	
	tree->height = height;
	tree->width = width;
	tree->channel = channel;
	tree->curSize = 0;

	//tree size estimation (TSE)
	nrmsd = 0;
	for (p = 0; p < numlevels; p++)
		nrmsd += ((double)dhist[p]) * ((double)dhist[p]);
	nrmsd = sqrt((nrmsd - (double)nredges) / ((double)nredges * ((double)nredges - 1.0)));
	if (mem_scheme == TSE)
		tree->maxSize = min(imgsize, (int32)(imgsize * A * (exp(SIGMA * nrmsd) + B + M)));
	else
		tree->maxSize = (int32)(2 * imgsize * size_init[mem_scheme]);

	Free(dhist);

	tree->parentAry = (int32*)Malloc((size_t)imgsize * sizeof(int32));
	tree->node = (AlphaNode*)Malloc((size_t)tree->maxSize * sizeof(AlphaNode));

	pParentAry = tree->parentAry;

#if DELAYED_ANODE_ALLOC
	levelroot[max_level + 1] = ANODE_CANDIDATE;
#else
	levelroot[max_level + 1] = NewAlphaNode(tree, (uint8)max_level);
#endif
	tree->node[levelroot[max_level + 1]].parentidx = levelroot[max_level + 1];

	current_level = max_level;
	x0 = imgsize >> 1;
	hqueue_push(hqueue, x0, current_level);

	iChild = levelroot[max_level + 1];
	while (current_level <= max_level)
	{
		while (hqueue->min_level <= current_level)
		{
			p = hqueue_pop(hqueue);
			if (is_visited(isVisited, p))
			{
				hqueue_find_min_level(hqueue);
				continue;
			}
			visit(isVisited, p);
#if !HQUEUE_COST_AMORTIZE
			hqueue_find_min_level();
#endif

			if (LEFT_AVAIL(p, width) && !is_visited(isVisited, p - 1))
			{
				dissim = (int32)dimg[dimg_idx_h(p - 1)];
				hqueue_push(hqueue, p - 1, dissim);
				if (levelroot[dissim] == NULL_LEVELROOT)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (RIGHT_AVAIL(p, width) && !is_visited(isVisited, p + 1))
			{
				dissim = (int32)dimg[dimg_idx_h(p)];
				hqueue_push(hqueue, p + 1, dissim);
				if (levelroot[dissim] == NULL_LEVELROOT)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (UP_AVAIL(p, width) && !is_visited(isVisited, p - width))
			{
				dissim = (int32)dimg[dimg_idx_v(p - width)];
				hqueue_push(hqueue, p - width, dissim);
				if (levelroot[dissim] == NULL_LEVELROOT)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (DOWN_AVAIL(p, width, imgsize) && !is_visited(isVisited, p + width))
			{
				dissim = (int32)dimg[dimg_idx_v(p)];
				hqueue_push(hqueue, p + width, dissim);
				if (levelroot[dissim] == NULL_LEVELROOT)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}

			if (current_level > hqueue->min_level)
				current_level = hqueue->min_level;
#if HQUEUE_COST_AMORTIZE
			else
				hqueue_find_min_level(hqueue);
#endif

#if DELAYED_ANODE_ALLOC
			connectPix2Node(tree, pParentAry, p, img[p], levelroot, current_level); 
#else
			connectPix2Node(pParentAry, p, img[p], tree->node + levelroot[current_level], levelroot[current_level]);
#endif

		}
//		if(tree->curSize > 22051838 && (tree->curSize))
	//		printf("curSize: %d\n",tree->curSize);
		//Redundant node removal
		if (tree->node[iChild].parentidx == levelroot[current_level] &&
			tree->node[levelroot[current_level]].area == tree->node[iChild].area)
		{
			levelroot[current_level] = iChild;
#if DELAYED_ANODE_ALLOC
			tree->curSize--;
#endif
		}

		next_level = current_level + 1;
		while (next_level <= max_level && (levelroot[next_level] == NULL_LEVELROOT))
			next_level++;
#if DELAYED_ANODE_ALLOC
		connectNode2Node(tree, levelroot, tree->node + levelroot[current_level], next_level);
#else
		connectNode2Node(tree->node + levelroot[next_level], levelroot[next_level], tree->node + levelroot[current_level]);
#endif

		iChild = levelroot[current_level];
		levelroot[current_level] = NULL_LEVELROOT;
		current_level = next_level;

	}
	for (p = 0; p < imgsize; p++)
	{
		if (tree->node[pParentAry[p]].level)//Singleton 0-CC
		{
			x0 = tree->curSize++;
			tree->node[x0].level = 0;
			tree->node[x0].area = 1;
			tree->node[x0].maxPix =
			tree->node[x0].minPix = img[p];
			tree->node[x0].sumPix = (double)img[p];
			tree->node[x0].parentidx = pParentAry[p];
			pParentAry[p] = x0;
		}
	}

	hqueue_free(hqueue);
	Free(dimg);
	Free(levelroot);
	Free(isVisited);
}


void BuildAlphaTree(AlphaTree* tree, pixel *img, int32 height, int32 width, int32 channel)
{
	Flood(tree, img, height, width, channel);
}

void DeleteAlphaTree(AlphaTree* tree)
{
	Free(tree->parentAry);
	Free(tree->node);
	Free(tree);
}

int main(int argc, char **argv)
{	
	AlphaTree *tree;
	int32 width, height, channel;
	int32 cnt = 0;
	ofstream f;
	ifstream fcheck;
	char in;
	int32 i,contidx;
	std::string path;
	uint8 testimg[4 * 4] = { 4, 4, 2, 0, 4, 1, 1, 0, 0, 3, 0, 0, 2, 2, 0, 5 };

	contidx = 0;
//	f.open("C:/Users/jwryu/RUG/2018/AlphaTree/AlphaTree_grey_Exp.dat", std::ofstream::app);
	fcheck.open(OUTPUT_FNAME);
	if (fcheck.good())
	{
		cout << "Output file \"" << OUTPUT_FNAME << "\" already exists. Overwrite? (y/n/a)";
		//cin >> in;
		in = 'y';
		if (in == 'a')
		{
			f.open(OUTPUT_FNAME, std::ofstream::app);
			cout << "Start from : ";
			cin >> contidx;
		}
		else if (in == 'y')
			f.open(OUTPUT_FNAME);
		else
			exit(-1);
	}
	else
		f.open(OUTPUT_FNAME);
	
	cnt = 0;
	for (mem_scheme = 0; mem_scheme < 4; mem_scheme++) // memory scheme loop (TSE, Max, Linear, Exp)
	{
#if RUN_TSE_ONLY
		if (mem_scheme > 0)
			break;
#endif
#if RUN_MAX_ONLY
	if (mem_scheme > 1)
			break;
	mem_scheme = 1;
#endif
		for (i = 0; i < 2; i++) // grey, colour loop
		{
			if (i == 0)
				path = INPUTIMAGE_DIR;
			else
				path = INPUTIMAGE_DIR_COLOUR;

			for (auto & p : std::experimental::filesystem::directory_iterator(path))
			{
				if (++cnt < contidx)
				{
					cout << cnt << ": " << p << endl;
					continue;
				}
				cv::String str1(p.path().string().c_str());
				cv::Mat cvimg;
				if (i == 0)
					cvimg = imread(str1, cv::IMREAD_GRAYSCALE);
				else
				{
					cvimg = imread(str1, cv::IMREAD_COLOR);
					cv::cvtColor(cvimg, cvimg, CV_BGR2GRAY);
				}

				/*
				cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
				cv::imshow("Display window", cvimg);                   // Show our image inside it.
				cv::waitKey(0);
				getc(stdin);
				*/

				height = cvimg.rows;
				width = cvimg.cols;
				channel = cvimg.channels();

				cout << cnt << ": " << str1 << ' ' << height << 'x' << width << endl;

				if (channel != 1)
				{
					cout << "input should be a greyscale image" << endl;
					getc(stdin);
					exit(-1);
				}

				double runtime, minruntime;
				for (int testrep = 0; testrep < REPEAT; testrep++)
				{
					memuse = max_memuse = 0;
					auto wcts = std::chrono::system_clock::now();

					tree = (AlphaTree*)Malloc(sizeof(AlphaTree));
					//		start = clock();
					BuildAlphaTree(tree, cvimg.data, height, width, channel);
					//BuildAlphaTree(tree, testimg, 4, 4, 1);

					std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
					runtime = wctduration.count();
					minruntime = testrep == 0 ? runtime : min(runtime, minruntime);

					if (testrep < (REPEAT - 1))
						DeleteAlphaTree(tree);
				}
				f << p.path().string().c_str() << '\t' << height << '\t' << width << '\t' << max_memuse << '\t' << nrmsd << '\t' << tree->maxSize << '\t' << tree->curSize << '\t' << minruntime << mem_scheme << i << endl;

				cout << "Time Elapsed: " << minruntime << endl;
				cvimg.release();
				str1.clear();
				DeleteAlphaTree(tree);
			}
		}
	}

	f.close();
	return 0;
}