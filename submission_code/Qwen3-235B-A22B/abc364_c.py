import sys

def main():
    n, X, Y = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check for k=1 case
    for i in range(n):
        if A[i] > X or B[i] > Y:
            print(1)
            return
    
    # Helper function to compute minimal k for a given sorted list
    def compute_k(sorted_pairs):
        sumA = 0
        sumB = 0
        for idx, (a, b) in enumerate(sorted_pairs):
            sumA += a
            sumB += b
            if sumA > X or sumB > Y:
                return idx + 1
        return float('inf')
    
    # Sort by A descending, then B descending
    sorted_a = sorted(zip(A, B), key=lambda x: (-x[0], -x[1]))
    ka = compute_k(sorted_a)
    
    # Sort by B descending, then A descending
    sorted_b = sorted(zip(A, B), key=lambda x: (-x[1], -x[0]))
    kb = compute_k(sorted_b)
    
    # Sort by sum A+B descending, then A and B descending
    sorted_ab = sorted(zip(A, B), key=lambda x: (-(x[0] + x[1]), -x[0], -x[1]))
    kab = compute_k(sorted_ab)
    
    ans = min(ka, kb, kab)
    if ans == float('inf'):
        print(n)
    else:
        print(ans)

if __name__ == "__main__":
    main()