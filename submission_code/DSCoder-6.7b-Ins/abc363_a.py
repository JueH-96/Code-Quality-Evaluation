# YOUR CODE HERE

def calculate_rating_increase(rating):
    if rating <= 99:
        return 99 - rating
    elif rating <= 199:
        return 199 - rating
    elif rating <= 299:
        return 299 - rating

rating = int(input())
print(calculate_rating_increase(rating))