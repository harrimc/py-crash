num_customers = [137, 147, 135, 128, 170, 174, 165, 146, 126, 159,
                 141, 148, 132, 147, 168, 153, 170, 161, 148, 152,
                 141, 151, 131, 149, 164, 163, 143, 143, 166, 171]

def avg_first_seven() :
    average = (sum(num_customers[0:7]))/7
    return average

def avg_last_seven() :
    lsev = (sum(num_customers[-7:]))/7
    return lsev

def max_month() :
    high = max(num_customers)
    return high
def min_month() :
    low = min(num_customers)
    return low
print(avg_first_seven())
print(avg_last_seven())
print(max_month())
print(min_month())
