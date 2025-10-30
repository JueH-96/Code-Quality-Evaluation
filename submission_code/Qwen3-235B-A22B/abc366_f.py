import sys

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    
    A = []
    B = []
    for _ in range(N):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        A.append(a)
        B.append(b)
    
    used = [False] * N
    current_value = 1
    
    for _ in range(K):
        best_val = -float('inf')
        best_idx = -1
        best_a = -1
        best_b = -1
        
        for i in range(N):
            if not used[i]:
                candidate = A[i] * current_value + B[i]
                if candidate > best_val:
                    best_val = candidate
                    best_idx = i
                    best_a = A[i]
                    best_b = B[i]
                elif candidate == best_val:
                    if B[i] > best_b:
                        best_val = candidate
                        best_idx = i
                        best_a = A[i]
                        best_b = B[i]
                    elif B[i] == best_b and A[i] > best_a:
                        best_val = candidate
                        best_idx = i
                        best_a = A[i]
                        best_b = B[i]
        
        if best_idx == -1:
            break
        
        used[best_idx] = True
        current_value = best_val
    
    print(current_value)

if __name__ == "__main__":
    main()