import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    
    total_sum = 0
    
    # We will query in powers of 2 ranges
    i = 0
    while (1 << i) <= (R - L + 1):
        step = 1 << i
        for j in range((L + step - 1) // step, (R + 1) // step):
            l = step * j
            r = min(step * (j + 1) - 1, R)
            if l > R or r < L:
                continue
            print(f"? {i} {j}")
            sys.stdout.flush()
            T = int(input().strip())
            if T == -1:
                return
            total_sum = (total_sum + T) % 100
        i += 1
    
    print(f"! {total_sum}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()