from typing import Union, List
from pydantic import BaseModel
from fastapi import Query
from typing import Optional
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str


class UserInDB(User):
    hashed_password: str


class spend_analysis_input(BaseModel):
    item_cat: str
    item_sec1: str
    item_sec2: str
    item: Optional[str] = None
    
class trend_analysis_input(BaseModel):
    item_cat: str
    item_sec1: str
    item_sec2: Optional[str] = None
    port: Optional[str] = None
   
class supplier_evaluation_input(BaseModel):
    item: str
    po_qty:int
    port:str

class demand_forecasting_input(BaseModel):
    # vessel_type: str
    # vessel_sub_type: str  
    vessel_name: str
    number_quarters: int
   


