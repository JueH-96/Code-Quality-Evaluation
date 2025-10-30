def main():
    import sys
    input_data = sys.stdin.read().strip()
    if input_data:
        N = int(input_data)
        # Create the alternating string:
        # "10" repeated N times gives us N ones and N zeros,
        # then adding one more "1" at the end gives exactly N+1 ones.
        result = "10" * N + "1"
        sys.stdout.write(result)

if __name__ == "__main__":
    main()