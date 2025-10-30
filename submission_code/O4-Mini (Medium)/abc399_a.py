def main():
    import sys
    data = sys.stdin.read().split()
    # data[0] is N, data[1] is S, data[2] is T
    S = data[1]
    T = data[2]
    # Count positions where characters differ
    answer = sum(1 for a, b in zip(S, T) if a != b)
    print(answer)

if __name__ == "__main__":
    main()