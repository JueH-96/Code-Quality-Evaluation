from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # We use DP where each state is represented by a tuple (first_char, last_char)
        # that describes the resulting string and its minimal achieved length.
        # dp[(a, b)] = minimal length of a string that starts with character a and ends with character b
        #
        # The join rules:
        # join(x, y) = x + y if x[-1] != y[0]
        #            = x + y[1:] if x[-1] == y[0]   (effectively subtract 1 from total length)
        #
        # At each step i (starting from i = 1) we can choose one of two operations:
        # 1) str_i = join(str_{i-1}, words[i]):
        #      new first = str_{i-1}[0], new last = words[i][-1]
        #      cost = dp[state] + len(words[i]) - (1 if str_{i-1}[-1] == words[i][0] else 0)
        # 2) str_i = join(words[i], str_{i-1}):
        #      new first = words[i][0], new last = str_{i-1}[-1]
        #      cost = dp[state] + len(words[i]) - (1 if words[i][-1] == str_{i-1}[0] else 0)
        #
        # Base case: start with the first word
        # dp[(words[0][0], words[0][-1])] = len(words[0])
        
        dp = {}
        first = words[0][0]
        last = words[0][-1]
        dp[(first, last)] = len(words[0])
        
        # Process remaining words
        for i in range(1, len(words)):
            word = words[i]
            newdp = {}
            for (cur_first, cur_last), cost in dp.items():
                # Option 1: join(current_string, word)
                discount = 1 if cur_last == word[0] else 0
                new_cost = cost + len(word) - discount
                state = (cur_first, word[-1])
                if state not in newdp or newdp[state] > new_cost:
                    newdp[state] = new_cost
                
                # Option 2: join(word, current_string)
                discount = 1 if word[-1] == cur_first else 0
                new_cost = cost + len(word) - discount
                state = (word[0], cur_last)
                if state not in newdp or newdp[state] > new_cost:
                    newdp[state] = new_cost
            
            dp = newdp
        
        # The answer is the minimal length among all states.
        return min(dp.values())


# The below code is provided for you to run tests if needed.
if __name__ == '__main__':
    # Example test cases
    sol = Solution()
    test_cases = [
        ["aa","ab","bc"],
        ["ab","b"],
        ["aaa","c","aba"]
    ]
    
    for words in test_cases:
        result = sol.minimizeConcatenatedLength(words)
        print(result)