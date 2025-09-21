def get_expected_cost(beds, baths) :
    Cost = ((beds*30000) + (baths*10000) + 80000)
    return Cost

print('To estimate your house cost we need to know 2 things:')
print('1. How many bedrooms do you have?')
bed = int(input('> '))

print('2. How many bathrooms do you have?')
bath = int(input('> '))

pri = get_expected_cost(bed, bath)
print('Your house price is', pri)