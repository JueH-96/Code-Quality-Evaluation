def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    dishes = data[1:]

    for i in range(1, N):
        if dishes[i] == "sweet" and dishes[i-1] == "sweet":
            # Found two consecutive sweet dishes
            # If this happens before the last dish, Takahashi cannot finish
            if i < N - 1:
                print("No")
                return
            else:
                # If this is on the last dish, he still finishes
                print("Yes")
                return

    # If no consecutive sweet pair was found (or only at the last dish)
    print("Yes")

if __name__ == "__main__":
    main()