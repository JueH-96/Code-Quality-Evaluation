def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    strings = data[1:]
    # Sort strings by their length (they are guaranteed distinct lengths)
    strings.sort(key=len)
    # Concatenate and print
    print("".join(strings))

if __name__ == "__main__":
    main()