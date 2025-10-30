import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr + N]))
        ptr += N
        
        SA = []
        sum_so_far = 0
        for a in A:
            sum_so_far += a
            SA.append(sum_so_far)
        
        S = SA[-1]
        possible = True
        sum_b = 0
        prev_b = 0
        
        for i in range(N):
            sa = SA[i]
            required = sa - sum_b
            current_b = max(prev_b, required)
            sum_b += current_b
            prev_b = current_b
            if sum_b > S:
                possible = False
                break
        
        if possible and sum_b == S:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()