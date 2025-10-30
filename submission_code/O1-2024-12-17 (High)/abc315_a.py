def main():
    S = input().strip()
    vowels = {'a','e','i','o','u'}
    result = ''.join(ch for ch in S if ch not in vowels)
    print(result)

main()