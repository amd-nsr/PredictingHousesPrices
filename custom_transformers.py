from	sklearn.base	import	BaseEstimator,	TransformerMixin
import numpy as np
from sklearn.preprocessing import LabelBinarizer 
rooms_ix,	bedrooms_ix,	population_ix,	household_ix	=	3,	4,	5,	6

class CombinedAttributesAdder(BaseEstimator,	TransformerMixin):
    def	__init__(self,	add_bedrooms_per_room = True):	#	no	*args	or	**kargs
        self.add_bedrooms_per_room	=	add_bedrooms_per_room
    def	fit(self,	X,	y=None):
        return self		#	nothing	else	to	do

    def	transform(self,	X,	y=None):
        rooms_per_household	=	X[:, rooms_ix]	/	X[:, household_ix]
        population_per_household	=	X[:, population_ix]	 /	X[:, household_ix]
        if	self.add_bedrooms_per_room:
            bedrooms_per_room	=	X[:, bedrooms_ix]	/	X[:, rooms_ix]
            return	np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return	np.c_[X, rooms_per_household, population_per_household]

#attr_adder	=	CombinedAttributesAdder(add_bedrooms_per_room=False)
#housing_extra_attribs	=	attr_adder.transform(housing.values)

class	DataFrameSelector(BaseEstimator,	TransformerMixin):
    def	__init__(self,	attribute_names):
        self.attribute_names	=	attribute_names
    def	fit(self,	X,	y=None):
        return	self
    def	transform(self,	X):
        return	X[self.attribute_names].values

class CustomLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)