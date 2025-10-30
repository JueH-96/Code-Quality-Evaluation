def main():
    import sys
    data = sys.stdin.read().split()
    N, T, A = map(int, data)
    # The result is already decided if one candidate has more than half the total votes.
    if 2 * max(T, A) > N:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()