def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # We need N zeros and N+1 ones, alternating, starting with '1'
    # So the pattern is '10' repeated N times, then a final '1'
    result = "10" * N + "1"
    print(result)

if __name__ == "__main__":
    main()