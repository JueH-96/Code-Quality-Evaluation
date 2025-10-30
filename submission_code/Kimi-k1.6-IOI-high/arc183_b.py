import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = int(input[ptr]), int(input[ptr+1])
        ptr +=2
        A = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        B = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        
        # Check if all elements in B are present in A
        a_elements = set(A)
        valid = True
        for x in B:
            if x not in a_elements:
                valid = False
                break
        if not valid:
            print("No")
            continue
        
        # Compute min and max indices for each element in A
        min_a = {}
        max_a = {}
        for i, x in enumerate(A):
            if x not in min_a:
                min_a[x] = i
            else:
                if i < min_a[x]:
                    min_a[x] = i
            if x not in max_a:
                max_a[x] = i
            else:
                if i > max_a[x]:
                    max_a[x] = i
        
        # Compute min and max indices for each element in B
        min_b = {}
        max_b = {}
        for i, x in enumerate(B):
            if x not in min_b:
                min_b[x] = i
            else:
                if i < min_b[x]:
                    min_b[x] = i
            if x not in max_b:
                max_b[x] = i
            else:
                if i > max_b[x]:
                    max_b[x] = i
        
        # Check conditions for each unique element in B
        unique_b = set(B)
        possible = True
        for x in unique_b:
            if x not in min_a or x not in max_a:
                possible = False
                break
            a_min = min_a[x]
            a_max = max_a[x]
            b_min = min_b[x]
            b_max = max_b[x]
            if b_min < a_min - K or b_max > a_max + K:
                possible = False
                break
        
        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()