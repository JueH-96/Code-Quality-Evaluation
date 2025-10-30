def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    takahashi_score = 0
    aoki_score = 0
    index = 1
    for _ in range(n):
        x = int(input_data[index])
        y = int(input_data[index+1])
        takahashi_score += x
        aoki_score += y
        index += 2
    if takahashi_score > aoki_score:
        print("Takahashi")
    elif aoki_score > takahashi_score:
        print("Aoki")
    else:
        print("Draw")

if __name__ == '__main__':
    main()