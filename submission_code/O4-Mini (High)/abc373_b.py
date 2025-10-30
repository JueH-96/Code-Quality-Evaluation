def main():
    S = input().strip()
    # Map each character to its 0-based position on the keyboard
    pos = {ch: i for i, ch in enumerate(S)}
    # Compute the total traveled distance for typing A->B->...->Z
    total = 0
    for prev, curr in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "BCDEFGHIJKLMNOPQRSTUVWXYZ"):
        total += abs(pos[curr] - pos[prev])
    print(total)

if __name__ == "__main__":
    main()