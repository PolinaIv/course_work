# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Template file used by ExpGenerator to generate the actual
permutations.py file by replacing $XXXXXXXX tokens with desired values.

This permutations.py file was generated by:
'/home/polina/.local/lib/python2.7/site-packages/nupic-0.5.3.dev0-py2.7.egg/nupic/swarming/exp_generator/ExpGenerator.pyc'
"""

import os

from nupic.swarming.permutationhelpers import *

# The name of the field being predicted.  Any allowed permutation MUST contain
# the prediction field.
# (generated from PREDICTION_FIELD)
predictedField = 'classification'




permutations = {
  'aggregationInfo': {   'days': 0,
    'fields': [],
    'hours': 0,
    'microseconds': 0,
    'milliseconds': 0,
    'minutes': 0,
    'months': 0,
    'seconds': 0,
    'weeks': 0,
    'years': 0},

  'modelParams': {
    

    'sensorParams': {
      'encoders': {
          u'field1': PermuteEncoder(maxval=100000.0, fieldName='field1', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field2': PermuteEncoder(maxval=100000.0, fieldName='field2', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field3': PermuteEncoder(maxval=100000.0, fieldName='field3', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field4': PermuteEncoder(maxval=100000.0, fieldName='field4', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field5': PermuteEncoder(maxval=100000.0, fieldName='field5', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field6': PermuteEncoder(maxval=100000.0, fieldName='field6', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field7': PermuteEncoder(maxval=100000.0, fieldName='field7', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field8': PermuteEncoder(maxval=100000.0, fieldName='field8', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field9': PermuteEncoder(maxval=100000.0, fieldName='field9', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field10': PermuteEncoder(maxval=100000.0, fieldName='field10', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field11': PermuteEncoder(maxval=100000.0, fieldName='field11', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field12': PermuteEncoder(maxval=100000.0, fieldName='field12', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field13': PermuteEncoder(maxval=100000.0, fieldName='field13', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field15': PermuteEncoder(maxval=100000.0, fieldName='field15', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field16': PermuteEncoder(maxval=100000.0, fieldName='field16', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field17': PermuteEncoder(maxval=100000.0, fieldName='field17', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field18': PermuteEncoder(maxval=100000.0, fieldName='field18', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field19': PermuteEncoder(maxval=100000.0, fieldName='field19', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field20': PermuteEncoder(maxval=100000.0, fieldName='field20', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field21': PermuteEncoder(maxval=100000.0, fieldName='field21', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field22': PermuteEncoder(maxval=100000.0, fieldName='field22', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field23': PermuteEncoder(maxval=100000.0, fieldName='field23', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  u'field24': PermuteEncoder(maxval=100000.0, fieldName='field24', w=21, clipInput=True, minval=-100000.0, encoderClass='ScalarEncoder', n=PermuteInt(22, 521), ),
  '_classifierInput': dict(maxval=2, classifierOnly=True, clipInput=True, minval=0, n=PermuteInt(28, 521), fieldname='classification', w=21, type='ScalarEncoder', ),
      },
    },

    'spParams': {
      
    },

    'tpParams': {
      
    },

    'clParams': {
        'alpha': PermuteFloat(0.0001, 0.1),

    },
  }
}


# Fields selected for final hypersearch report;
# NOTE: These values are used as regular expressions by RunPermutations.py's
#       report generator
# (fieldname values generated from PERM_PREDICTED_FIELD_NAME)
report = [
          '.*classification.*',
         ]

# Permutation optimization setting: either minimize or maximize metric
# used by RunPermutations.
# NOTE: The value is used as a regular expressions by RunPermutations.py's
#       report generator
# (generated from minimize = "multiStepBestPredictions:multiStep:errorMetric='altMAPE':steps=\[0\]:window=1000:field=classification")
minimize = "multiStepBestPredictions:multiStep:errorMetric='altMAPE':steps=\[0\]:window=1000:field=classification"

minParticlesPerSwarm = 5

inputPredictedField = 'no'







maxModels = 200



def permutationFilter(perm):
  """ This function can be used to selectively filter out specific permutation
  combinations. It is called by RunPermutations for every possible permutation
  of the variables in the permutations dict. It should return True for valid a
  combination of permutation values and False for an invalid one.

  Parameters:
  ---------------------------------------------------------
  perm: dict of one possible combination of name:value
        pairs chosen from permutations.
  """

  # An example of how to use this
  #if perm['__consumption_encoder']['maxval'] > 300:
  #  return False;
  #
  return True