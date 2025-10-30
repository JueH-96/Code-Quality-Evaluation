import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        K = int(input[ptr+1])
        ptr += 2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Check if all elements in B are present in A
        setA = set(A)
        setB = set(B)
        if not setB.issubset(setA):
            print("No")
            continue
        
        # Proceed with sliding window
        a_freq = defaultdict(int)
        # Initialize the first window [0, K]
        for j in range(K + 1):
            a_freq[A[j]] += 1
        
        b_freq = defaultdict(int)
        possible = True
        for i in range(N):
            current_b = B[i]
            # Check if current_b is present in a_freq or b_freq
            if a_freq.get(current_b, 0) + b_freq.get(current_b, 0) == 0:
                possible = False
                break
            
            # Remove A[i] from a_freq
            a_freq[A[i]] -= 1
            if a_freq[A[i]] == 0:
                del a_freq[A[i]]
            
            # Add next element to a_freq if within bounds
            next_j = i + K + 1
            if next_j < N:
                a_freq[A[next_j]] += 1
            
            # Add current_b to b_freq
            b_freq[current_b] += 1
        
        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()