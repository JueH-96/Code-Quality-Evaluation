def can_color_all_black(N, K):
    if K == N // 2:
        return "No"
    else:
        return "Yes"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    index = 1
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        print(can_color_all_black(N, K))

if __name__ == "__main__":
    main()