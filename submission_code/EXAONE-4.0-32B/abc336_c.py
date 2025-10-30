def main():
    n_val = int(input().strip())
    n_val -= 1
    mapping_arr = ['0', '2', '4', '6', '8']
    digits = []
    num = n_val
    while num:
        num, r = divmod(num, 5)
        digits.append(r)
    if not digits:
        digits = [0]
    digits.reverse()
    res = ''.join(mapping_arr[d] for d in digits)
    print(res)

if __name__ == '__main__':
    main()