def can_form_palindrome(s, k):
    from collections import Counter
    
    char_count = Counter(s)
    odd_counts = sum(1 for count in char_count.values() if count % 2 != 0)
    
    if odd_counts <= k:
        return "YES"
    elif odd_counts - k <= 1:
        return "YES"
    else:
        return "NO"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        n, k = int(data[index]), int(data[index + 1])
        index += 2
        s = data[index]
        index += 1
        results.append(can_form_palindrome(s, k))
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()