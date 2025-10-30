def get_top(arr, K):
    indexed = sorted(enumerate(arr), key=lambda x: (-x[1], x[0]))
    return indexed[:K]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    L = int(data[ptr])
    ptr += 1
    
    a = list(map(int, data[ptr:ptr+N]))
    ptr += N
    
    b = list(map(int, data[ptr:ptr+M]))
    ptr += M
    
    forbidden = set()
    for _ in range(L):
        c = int(data[ptr]) - 1
        ptr += 1
        d = int(data[ptr]) - 1
        ptr += 1
        forbidden.add((c, d))
    
    top_a = get_top(a, 200)
    top_b = get_top(b, 200)
    
    max_sum = 0
    for a_idx, a_val in top_a:
        for b_idx, b_val in top_b:
            if (a_idx, b_idx) not in forbidden:
                current = a_val + b_val
                if current > max_sum:
                    max_sum = current
    print(max_sum)

if __name__ == "__main__":
    main()