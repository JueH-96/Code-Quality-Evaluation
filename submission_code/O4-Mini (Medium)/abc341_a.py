def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # We need N zeros and N+1 ones, alternating, starting and ending with '1'.
    # A concise way is to repeat "10" N times (which gives N '1's and N '0's),
    # then append one more '1' to have N+1 '1's.
    result = "10" * N + "1"
    sys.stdout.write(result)

if __name__ == "__main__":
    main()