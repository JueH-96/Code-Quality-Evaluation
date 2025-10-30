class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        current_streak = 0
        output = []
        for i in range(len(words)):
            word = words[i]
            if word == 'prev':
                if i == 0:
                    current_streak = 1
                else:
                    if words[i-1] == 'prev':
                        current_streak += 1
                    else:
                        current_streak = 1
                reverse_nums = nums[::-1]
                if (current_streak - 1) < len(reverse_nums):
                    output.append(reverse_nums[current_streak - 1])
                else:
                    output.append(-1)
            else:
                nums.append(int(word))
        return output