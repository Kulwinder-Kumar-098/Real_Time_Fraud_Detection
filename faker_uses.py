import pandas as pd
import random
from datetime import datetime
from faker import Faker
import uuid
# Configuration for mock data options
fake=Faker()
def random_genrator():
    user_ids=fake.first_name()+str(random.randint(100,999))
    amount=random.randint(1000,100000)
    timestamp=datetime.now()
    transaction_id=uuid.uuid1()
    merchants = random.choice(['Dmart', 'Punjab stores', 'ANAND GROCERY STORE', 'Aggarwal Super Market', 'Sham Super Store'])
    payment_methods = random.choice(['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash'])
    devices = random.choice(['Android Mobile', 'iOS Mobile', 'Desktop', 'POS Terminal'])
    locations = random.choice(['Industrial Area, Mohali', 'Sector 71, Mohali', 'Sector 79, Mohali', 'Sector 70, Mohali', 'Sector 91, Mohali'])
    return {
        'user_id': user_ids,
        'amount': amount,
        'timestamp': timestamp,
        'transaction_id': transaction_id,
        'merchant': merchants,
        'payment_method': payment_methods,
        'device': devices,
        'location': locations
    }
transaction_data = random_genrator()
df=pd.DataFrame([transaction_data])
print(df)