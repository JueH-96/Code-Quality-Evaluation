def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N, Q = int(input[ptr]), int(input[ptr+1])
    ptr +=2
    C = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    colors_in_box = [set() for _ in range(N+1)]  # 1-based
    for i in range(N):
        colors_in_box[i+1].add(C[i])
    for _ in range(Q):
        a = int(input[ptr])
        b = int(input[ptr+1])
        ptr +=2
        if not colors_in_box[a]:
            print(len(colors_in_box[b]))
        else:
            colors_in_box[b].update(colors_in_box[a])
            colors_in_box[a].clear()
            print(len(colors_in_box[b]))

if __name__ == "__main__":
    main()