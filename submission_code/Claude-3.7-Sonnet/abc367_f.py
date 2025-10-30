from collections import Counter

def main():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    for _ in range(Q):
        l, r, L, R = map(int, input().split())
        
        # Check if the length of the subsequences is the same
        if (r - l + 1) != (R - L + 1):
            print("No")
            continue
        
        # Extract subsequences (convert from 1-indexed to 0-indexed)
        a_subsequence = A[l-1:r]
        b_subsequence = B[L-1:R]
        
        # Check if the subsequence from A can be rearranged to match the subsequence from B
        if Counter(a_subsequence) == Counter(b_subsequence):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()