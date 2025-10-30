def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    x = int(data)
    # Compute ceil(x/10) using integer arithmetic
    result = -(-x // 10)
    print(result)

main()