import numpy
from scipy.special import comb, perm, factorial, factorial2
from scipy.stats import mode, entropy
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, dia_matrix
import string
from collections import Counter, deque, defaultdict
import math
from itertools import permutations, combinations, chain, combinations_with_replacement, accumulate, groupby
from bisect import bisect_left, bisect_right, insort_left, insort_right
import heapq
import sys
from copy import copy, deepcopy
from sys import stdin, stdout

def input():
    return sys.stdin.readline().strip()

def fastio():
    sys.stdin = open('input.txt', 'r')
    sys.stdout = open('output.txt', 'w')

INF = float('inf')
MOD = 1000000007
# fastio()
sys.setrecursionlimit(1000000)

n = int(input())
garbage_info = [tuple(map(int, input().split())) for _ in range(n)]

q = int(input())
for i in range(q):
    t, d = map(int, input().split())
    q, r = garbage_info[t-1]
    remaining = (r - (d % q)) % q
    print(d + remaining)