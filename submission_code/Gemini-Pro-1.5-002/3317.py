class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        n = len(words)
        count = Counter()
        for word in words:
            count += Counter(word)
        
        ans = 0
        odd = []
        for word in words:
            temp_count = Counter(word)
            flag = True
            odd_char = ''
            for char in temp_count:
                if temp_count[char] % 2 == 1:
                    if odd_char == '':
                        odd_char = char
                    elif odd_char != char:
                        flag = False
                        break
            if flag:
                if odd_char == '':
                    ans += 1
                    for char in temp_count:
                        count[char] -= temp_count[char]
                else:
                    odd.append((word, odd_char))
        
        i = 0
        while i < len(odd):
            word1, odd_char1 = odd[i]
            temp_count1 = Counter(word1)
            j = i + 1
            while j < len(odd):
                word2, odd_char2 = odd[j]
                temp_count2 = Counter(word2)
                if odd_char1 == odd_char2 and count[odd_char1] >= 2:
                    ans += 2
                    for char in temp_count1:
                        count[char] -= temp_count1[char]
                    for char in temp_count2:
                        count[char] -= temp_count2[char]
                    i = j + 1
                    break
                j += 1
            if j == len(odd):
                if count[odd_char1] >= 2:
                    ans += 1
                    for char in temp_count1:
                        count[char] -= temp_count1[char]
                i += 1
        
        for char in count:
            if count[char] >= 2:
                ans += count[char] // 2
        
        return ans