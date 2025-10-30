import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N = int(input[idx])
        idx += 1
        A = input[idx]
        idx += 1
        B = input[idx]
        idx += 1
        
        A_list = []
        for i in range(N):
            if A[i] == '1':
                A_list.append(i + 1)
        B_list = []
        for i in range(N):
            if B[i] == '1':
                B_list.append(i + 1)
        
        if len(A_list) < len(B_list):
            print(-1)
            continue
        
        A_sorted = sorted(A_list)
        B_sorted = sorted(B_list)
        m = len(B_sorted)
        max_dist = 0
        for a, b in zip(A_sorted[:m], B_sorted):
            current = abs(a - b)
            if current > max_dist:
                max_dist = current
        print(max_dist)
        
if __name__ == "__main__":
    main()