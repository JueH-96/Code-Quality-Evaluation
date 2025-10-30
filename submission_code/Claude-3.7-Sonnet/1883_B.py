# YOUR CODE HERE
def can_form_palindrome_after_removal(s, k):
    # Count the frequency of each character
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Count the number of characters with odd frequencies
    odd_count = sum(1 for count in char_count.values() if count % 2 != 0)
    
    # Check if we can reduce the odd count to at most 1 by removing k characters
    return odd_count - k <= 1

def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        s = input().strip()
        print("YES" if can_form_palindrome_after_removal(s, k) else "NO")

if __name__ == "__main__":
    main()