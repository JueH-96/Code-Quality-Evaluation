import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    # Precompute the marks array
    marks = []
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            marks.append(1)
        else:
            marks.append(0)
    
    # Compute prefix sum
    prefix_sum = [0] * (len(marks) + 1)
    current = 0
    for i in range(len(marks)):
        current += marks[i]
        prefix_sum[i + 1] = current
    
    # Process each query
    for _ in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        if l >= r:
            print(0)
        else:
            res = prefix_sum[r - 1] - prefix_sum[l - 1]
            print(res)

if __name__ == "__main__":
    main()