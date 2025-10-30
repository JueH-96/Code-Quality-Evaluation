def main():
    x_str = input().strip()
    x_float = float(x_str)
    # Format with up to 3 decimals, then strip trailing zeros and possible trailing dot
    result = f"{x_float:.3f}".rstrip('0').rstrip('.')
    print(result)

if __name__ == "__main__":
    main()