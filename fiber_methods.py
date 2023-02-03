from concurrent.futures import ThreadPoolExecutor
import sys
from itertools import product, permutations, chain
import numpy as np
import cv2
import scipy
from scipy import ndimage
from scipy.spatial import Delaunay
from skimage.morphology import label, skeletonize
from skimage import measure
from sklearn.neighbors import NearestNeighbors
import os
import math
import random
import h5py
from PIL import Image, ImageDraw, ImageOps
import networkx as nx
import mahotas as mh

def eucdist(x,y):
    return ((x[0] - x[1])**2 + (y[0] - y[1])**2)**.5


def cart2pol(x, y):
    '''Cartesian to polar conversion'''
    
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def hog_features(img, nbins = 64):
    '''Histogram of oriented gradients'''
    if len(img.shape) == 3:
        raise Exception("images must be 2D arrays")
        
    if np.sum(img) == 0:
        return [None] * 5
    
    img = img / img.max()
    dx = ndimage.sobel(img, 0)
    dy = ndimage.sobel(img, 1)
    rho, phi = cart2pol(dx, dy)
    rho = rho.flatten()
    phi = phi.flatten()
    phi[phi < 0] += np.pi
    it = np.pi / nbins
    bins = np.arange(0, np.pi, it).tolist()
    bin_indices = np.digitize(phi, bins)
    mag = np.bincount(bin_indices, weights = np.abs(rho)).tolist()
    if len(mag) <= nbins:
        mag = mag + [0]*(nbins - len(mag) + 1)
        
    last_bin_contribution = (phi - bins[-1]) / it * np.abs(rho)
    mag[0] += np.sum(last_bin_contribution[bin_indices == len(bins)])
    mag = mag[:-1]
    ang = np.array(bins)
    if np.sum(mag) == 0:
        return [None] * 5
    
    cosa = (mag * np.cos(ang))[mag != 0]
    sina = (mag * np.sin(ang))[mag != 0]
    LDM = np.arctan(np.sum(sina) / np.sum(cosa)) #Linear Directional Mean
    CV = 1. - ((np.sum(sina)**2 + np.sum(cosa)**2)**0.5) / np.sum(mag) #Circular Variance
    CSD = (-2 * np.log(1. - CV))**0.5 #Circular Standard Deviation
    
    return [LDM, CV, CSD, np.mean(mag[mag != 0]), np.std(mag[mag != 0])]


def count_uniques(Aa):
    dt = np.dtype((np.void, Aa.dtype.itemsize * Aa.shape[1]))
    b = np.ascontiguousarray(Aa).view(dt)
    unq, cnt = np.unique(b, return_counts = True)
    
    return cnt


def fractal_features(img):
    if len(img.shape) == 3:
        raise Exception("images must be 2D arrays")
        
    if np.sum(img) == 0:
        return [None, None]
    
    y, x = np.nonzero(img)
    coords = np.column_stack([y, x])

    ylim, xlim = img.shape

    # computing the fractal dimension
    scales = np.array([2**s for s in range(int(np.log2(np.min(img.shape))), -1, -1)])
    Ns=[]
    Ll = []
    
    # looping over several scales
    Hs = []
    for scale in scales:
        # computing the histogram
        H, edges = np.histogramdd(coords, bins = (np.arange(0, xlim + scale, scale),
                                                  np.arange(0, ylim + scale, scale)))
        
        Ns.append(np.sum(H > 0))
        Ll.append(np.std(H[H > 0]) / (np.mean(H[H > 0]) + sys.float_info.epsilon))
        
    coeffs = np.polyfit(np.log2(scales),
                        np.log2(Ns), 1)
    return [-coeffs[0], np.mean(Ll)]


def drawhex(w = 260, fill = 1, inv = 0):
    h = int(np.sqrt(3) * .5 * w/2) * 2
    polygon = [(0, h / 2), (w / 4, 0),(3 * w / 4, 0), 
               (w, h / 2), (3 * w / 4, h), (w / 4, h)]
    
    img_ = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img_).polygon(polygon, outline = 1, fill = fill)
    img = np.array(img_)
    pw = int((w - h) / 2)
    img = np.pad(img, ((pw, pw), (0, 0)), mode = 'constant', constant_values = (0))
    if inv != 0:
        img = ImageOps.invert(img_)
        img = np.array(img) / 255.
    
    return img.astype(int)


# get neighbors d index
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype = np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1, ) + shape[::-1][:-1])
    
    return np.dot(idx, acc[::-1])


# my mark
def mark(img): # mark the array use (0, 1, 2)
    nbs = neighbors(img.shape)
    img = img.ravel()
    for p in range(len(img)):
        if img[p] == 0:
            continue
            
        s = 0
        for dp in nbs:
            if img[p + dp] != 0:
                s += 1
                
        if s == 2:
            img[p] = 1
            
        else:
            img[p] = 2
            

# trans index to r, c...
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype = np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i] // acc[j]
            idx[i] -= rst[i,j] * acc[j]
            
    rst -= 1
    
    return rst


# fill a node (may be two or more points)
def fill(img, p, num, nbs, acc, buf):
    back = img[p]
    img[p] = num
    buf[0] = p
    cur = 0; s = 1;
    
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if img[cp] == back:
                img[cp] = num
                buf[s] = cp
                s += 1
                
        cur += 1
        if cur==s:
            break
            
    return idx2rc(buf[:s], acc)


# trace the edge and use a buffer, then buf.copy, if use [] numba not works
def trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0;
    newp = 0
    cur = 0

    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:c1=img[cp]
                else: c2 = img[cp]
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, idx2rc(buf[:cur], acc))


# parse the image then get the nodes and edges
def parse_struc(img):
    nbs = neighbors(img.shape)
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    pts = np.array(np.where(img == 2))[0]
    buf = np.zeros(131072, dtype = np.int64) # megabyte
    num = 10
    nodes = []
    for p in pts:
        if img[p] == 2:
            nds = fill(img, p, num, nbs, acc, buf)
            num += 1
            nodes.append(nds)

    edges = []
    for p in pts:
        for dp in nbs:
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
                
    return nodes, edges


# use nodes and edges build a networkx graph
def build_graph(nodes, edges, multi = False):
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts = nodes[i], o = nodes[i].mean(axis = 0))
        
    for s, e, pts in edges:
        l = np.linalg.norm(pts[1:] - pts[:-1], axis = 1).sum()
        graph.add_edge(s,e, pts = pts, weight = l)
        
    return graph


def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape) + 2), dtype = np.uint16)
    buf[tuple([slice(1, -1)] * buf.ndim)] = ske
    
    return buf


def build_sknw(ske, multi = False):
    buf = buffer(ske)
    mark(buf)
    nodes, edges = parse_struc(buf)
    
    return build_graph(nodes, edges, multi)
    
# draw the graph
def draw_graph(img, graph, cn = 255, ce = 128):
    acc = np.cumprod((1, ) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for idx in graph.nodes():
        pts = graph.node[idx]['pts']
        img[np.dot(pts, acc)] = cn
        
    for (s, e) in graph.edges():
        eds = graph[s][e]
        
        for i in eds:
            pts = eds[i]['pts']
            img[np.dot(pts, acc)] = ce

            
def graph_distances(graph):
    e = []
    for g in graph.nodes():
        e.append(graph._node[g]['o'])
        
    e = np.array(e)
    vmat = []
    for (st, en) in graph.edges():
        vmat.append([st, en])
        
    d = []
    for i in range(len(vmat)):
        d.append(eucdist(e[vmat[i]][:, 0], e[vmat[i]][:, 1]))
        
    return d


def morphometry_features(img):
    if len(img.shape) == 3:
        raise Exception("images must be 2D arrays")
    
    strct = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])
    
    lab,_ = ndimage.label(img, structure = strct)
    objs = ndimage.find_objects(lab)
    if len(objs) == 0:
        
        return [None] * 17
    
    FL = []
    FP = []
    FS = []
    FW = []
    FD = []
    nFD = []
    
    for n, ob in enumerate(objs):
        #prepare image data
        if type(ob).__name__ == 'NoneType':
            continue
        
        msk = lab[ob[0].start : ob[0].stop, ob[1].start : ob[1].stop].copy()
        msk[msk != n + 1] = 0
        
        skel = skeletonize(msk.astype(np.float) / (n + 1))
        skel = skel * 1
        skel = np.pad(skel, ((1, 1), (1, 1)),
                      'constant', constant_values = 0)
        
        s = np.moveaxis(np.array(np.nonzero(skel)), -1, 0)
        
        #fiber length
        graph = build_sknw(skel)
        FL.append(np.sum(graph_distances(graph)))
        
        #fiber path
        msk = np.pad(msk, ((1, 1), (1, 1)), 
                     'constant', constant_values = 0)
        
        CNT = np.row_stack(measure.find_contours(msk, .8))
        FP.append(float(CNT.shape[0]) / 2)
        
        #fiber width
        nbrs = NearestNeighbors(n_neighbors = 2, 
                                algorithm = 'ball_tree').fit(CNT)
        w, _ = nbrs.kneighbors(s)
        FW.append(np.median(w) * 2)
        
        #fiber straightness
        FS.append(FL[-1] / FP[-1])
        
    #fiber density by bulk pixels
    FD = []
    if np.max(img != 0):
        FD.append(np.sum(img) / np.max(img))
    
    #fiber density by endpoints
    graph2 = build_sknw(skeletonize(img / 255))
    e = []
    for g in graph2.nodes():
        e.append(graph2._node[g]['o'])
        
    e = np.array(e)
    
    if e.shape[0] < 2:
        nFD = [e.shape[0], None, None, None]
        
    else:
        nbrs = NearestNeighbors(n_neighbors = e.shape[0],
                                algorithm = 'ball_tree').fit(e)
        distances, indices = nbrs.kneighbors(e)
        nFD = [e.shape[0],
               np.mean(distances[:,1:]),
               np.median(distances[:,1:]),
               np.std(distances[:,1:])]
        
    #aggregate
    FL = [np.mean(FL), np.median(FL), np.std(FL)]
    FP = [np.mean(FP), np.median(FP), np.std(FP)]
    FS = [np.mean(FS), np.median(FS), np.std(FS)]
    FW = [np.mean(FW), np.median(FW), np.std(FW)]
    
    
    return FL + FP + FS + FW + FD + nFD

def texture_features(img):
    if len(img.shape) == 3:
        raise Exception("images must be 2D arrays")
        
    if np.sum(img) != 0:
        
        return mh.features.haralick(img).mean(0).tolist()
    
    else:
        
        return [None] * 13

