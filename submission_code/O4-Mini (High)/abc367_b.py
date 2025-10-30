def main():
    s = input().strip()
    # Remove trailing zeros in the fractional part
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    print(s)

main()