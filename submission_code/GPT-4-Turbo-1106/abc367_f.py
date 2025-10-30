from collections import Counter

def can_rearrange_subsequence(A, B, queries):
    for l, r, L, R in queries:
        subseq_A = A[l-1:r]
        subseq_B = B[L-1:R]
        if Counter(subseq_A) == Counter(subseq_B):
            print("Yes")
        else:
            print("No")

def main():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    queries = [tuple(map(int, input().split())) for _ in range(Q)]
    can_rearrange_subsequence(A, B, queries)

if __name__ == "__main__":
    main()