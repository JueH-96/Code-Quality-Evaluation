def main():
    import sys
    input = sys.stdin.readline
    S = input().strip()
    answer = "Yes"
    # Check positions: i = 2, 4, 6, ... 16 which correspond to indices 1, 3, ... 15 (0-indexed)
    for i in range(1, 16, 2):
        if S[i] != '0':
            answer = "No"
            break
    print(answer)

if __name__ == '__main__':
    main()