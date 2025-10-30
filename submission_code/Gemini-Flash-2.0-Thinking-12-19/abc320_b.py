def is_palindrome(s):
  return s == s[::-1]

def solve():
  s = input()
  max_len = 0
  for i in range(len(s)):
    for j in range(i, len(s)):
      sub = s[i:j+1]
      if is_palindrome(sub):
        max_len = max(max_len, len(sub))
  print(max_len)

if __name__ == "__main__":
  solve()