import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        valid = True
        K_list = []
        for __ in range(N):
            A = int(data[idx])
            B = int(data[idx+1])
            C = int(data[idx+2])
            idx +=3
            if C <= B:
                valid = False
            K = (C - B - 1) // A
            K_list.append(K)
        if not valid:
            print(0)
            continue
        X_max = min(K_list)
        if X_max < 1:
            print(0)
            continue
        sum_total = 0
        for x in range(1, X_max + 1):
            min_f = float('inf')
            for __ in range(N):
                A = int(data[idx])
                B = int(data[idx+1])
                C = int(data[idx+2])
                idx +=3
                numerator = C - A * x - 1
                f = numerator // B
                if f < min_f:
                    min_f = f
            if min_f >= 1:
                sum_total += min_f
        print(sum_total)
        
if __name__ == '__main__':
    main()