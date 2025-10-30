class Solution:
    def findLatestTime(self, s: str) -> str:
        def is_valid_time(time: str) -> bool:
            """Check if a given time string is a valid 12-hour format time."""
            hours, minutes = time.split(":")
            return 0 <= int(hours) <= 11 and 0 <= int(minutes) <= 59

        def get_largest_digit(index: int, time: str) -> str:
            """Get the largest possible digit for a given index in the time string."""
            if index == 0:
                # The first digit of the hour can be at most 1
                return "1" if time[index] == "?" else time[index]
            elif index == 1:
                # The second digit of the hour can be at most 9 if the first digit is 0 or 1, otherwise it's at most 3
                if time[index] == "?":
                    return "9" if time[0] in ["0", "1", "?"] else "3"
                else:
                    return time[index]
            elif index == 3:
                # The first digit of the minute can be at most 5
                return "5" if time[index] == "?" else time[index]
            else:
                # The second digit of the minute can be at most 9
                return "9" if time[index] == "?" else time[index]

        # Start with the input string
        time = list(s)

        # Replace the "?" characters from right to left to get the latest possible time
        for i in range(len(time) - 1, -1, -1):
            if time[i] == "?":
                time[i] = get_largest_digit(i, "".join(time))

        # Join the characters back into a string and return the result
        return "".join(time)