def main():
    import sys
    s = sys.stdin.readline().strip()
    # Split on '|' and ignore the empty first and last parts
    parts = s.split('|')
    # parts[1:-1] are the segments of '-' whose lengths are the A_i
    result = [str(len(segment)) for segment in parts[1:-1]]
    print(" ".join(result))

if __name__ == "__main__":
    main()