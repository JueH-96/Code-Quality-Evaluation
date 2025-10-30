def main():
    import sys
    input_data = sys.stdin.read().split()
    if len(input_data) < 2:
        return
    S = input_data[0].strip()
    T = input_data[1].strip()

    # Function to check if 'target' is a subsequence of S (by converting S's characters to uppercase)
    def is_subsequence(S, target):
        i = 0
        for char in S:
            if char.upper() == target[i]:
                i += 1
                if i == len(target):
                    return True
        return False

    # First check: can we pick a length-3 subsequence?
    if is_subsequence(S, T):
        sys.stdout.write("Yes")
        return

    # Second check: can we pick a length-2 subsequence and append X?
    if T[2] == 'X' and is_subsequence(S, T[:2]):
        sys.stdout.write("Yes")
        return

    sys.stdout.write("No")

if __name__ == '__main__':
    main()