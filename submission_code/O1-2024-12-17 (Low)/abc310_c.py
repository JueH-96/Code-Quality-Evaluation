def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    sticks = input_data[1:]

    seen = set()
    for s in sticks:
        rev_s = s[::-1]
        canonical = min(s, rev_s)
        seen.add(canonical)

    print(len(seen))

# Do not remove the call to main()
main()