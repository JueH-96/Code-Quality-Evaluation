from decimal import Decimal, getcontext

getcontext().prec = 30
ln2 = Decimal(2).ln()

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx + n]))
        idx += n
        groups = {}
        total = 0
        for num in a:
            ln_num = Decimal(str(num)).ln()
            c = ln_num - Decimal(str(num)) * ln2
            c_rounded = c.quantize(Decimal('1.000000000000000'))
            key = format(c_rounded, 'f')
            if '.' in key:
                integer_part, fractional_part = key.split('.')
                fractional_part = fractional_part.ljust(15, '0')[:15]
                key = f"{integer_part}.{fractional_part}"
            else:
                key = f"{key}.000000000000000"
            if key in groups:
                groups[key] += 1
            else:
                groups[key] = 1
        total = 0
        for count in groups.values():
            if count >= 2:
                total += count * (count - 1) // 2
        print(total)

if __name__ == '__main__':
    main()