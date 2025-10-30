import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    r = min(k, n - k)
    if r == 0:
        print(0)
        return
        
    if k > n - k:
        total_xor = 0
        for a in A:
            total_xor ^= a
        best = 0
        if r == 1:
            for i in range(n):
                candidate = total_xor ^ A[i]
                if candidate > best:
                    best = candidate
        elif r == 2:
            for i in range(n):
                for j in range(i+1, n):
                    xor_val = A[i] ^ A[j]
                    candidate = total_xor ^ xor_val
                    if candidate > best:
                        best = candidate
        elif r == 3:
            for i in range(n):
                for j in range(i+1, n):
                    for k_index in range(j+1, n):
                        xor_val = A[i] ^ A[j] ^ A[k_index]
                        candidate = total_xor ^ xor_val
                        if candidate > best:
                            best = candidate
        elif r == 4:
            for i in range(n):
                for j in range(i+1, n):
                    for k_index in range(j+1, n):
                        for l in range(k_index+1, n):
                            xor_val = A[i] ^ A[j] ^ A[k_index] ^ A[l]
                            candidate = total_xor ^ xor_val
                            if candidate > best:
                                best = candidate
        elif r == 5:
            for i in range(n):
                for j in range(i+1, n):
                    for k_index in range(j+1, n):
                        for l in range(k_index+1, n):
                            for m in range(l+1, n):
                                xor_val = A[i] ^ A[j] ^ A[k_index] ^ A[l] ^ A[m]
                                candidate = total_xor ^ xor_val
                                if candidate > best:
                                    best = candidate
        print(best)
    else:
        best = 0
        if r == 1:
            for i in range(n):
                if A[i] > best:
                    best = A[i]
        elif r == 2:
            for i in range(n):
                for j in range(i+1, n):
                    xor_val = A[i] ^ A[j]
                    if xor_val > best:
                        best = xor_val
        elif r == 3:
            for i in range(n):
                for j in range(i+1, n):
                    for k_index in range(j+1, n):
                        xor_val = A[i] ^ A[j] ^ A[k_index]
                        if xor_val > best:
                            best = xor_val
        elif r == 4:
            for i in range(n):
                for j in range(i+1, n):
                    for k_index in range(j+1, n):
                        for l in range(k_index+1, n):
                            xor_val = A[i] ^ A[j] ^ A[k_index] ^ A[l]
                            if xor_val > best:
                                best = xor_val
        elif r == 5:
            for i in range(n):
                for j in range(i+1, n):
                    for k_index in range(j+1, n):
                        for l in range(k_index+1, n):
                            for m in range(l+1, n):
                                xor_val = A[i] ^ A[j] ^ A[k_index] ^ A[l] ^ A[m]
                                if xor_val > best:
                                    best = xor_val
        print(best)

if __name__ == "__main__":
    main()