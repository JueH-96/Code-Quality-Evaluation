def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    dishes = data[1:]

    # We'll check for consecutive sweet dishes.
    for i in range(N - 1):
        if dishes[i] == "sweet" and dishes[i+1] == "sweet":
            # If these consecutive sweets are NOT the last pair, he can't finish.
            if i+1 < N - 1:
                print("No")
                return
            # If it's the last pair, he's already eaten all dishes.
            else:
                print("Yes")
                return

    # If we never found consecutive sweets or found only at last pair, print Yes.
    print("Yes")

if __name__ == "__main__":
    main()