��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8��
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:w*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:w*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:w*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:w*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ww*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:ww*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:w*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:w*
dtype0
z
output_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:w* 
shared_nameoutput_4/kernel
s
#output_4/kernel/Read/ReadVariableOpReadVariableOpoutput_4/kernel*
_output_shapes

:w*
dtype0
r
output_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput_4/bias
k
!output_4/bias/Read/ReadVariableOpReadVariableOpoutput_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:w*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:w*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:w*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:w*
dtype0
�
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ww*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:ww*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:w*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:w*
dtype0
�
Adam/output_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:w*'
shared_nameAdam/output_4/kernel/m
�
*Adam/output_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_4/kernel/m*
_output_shapes

:w*
dtype0
�
Adam/output_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output_4/bias/m
y
(Adam/output_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_4/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:w*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:w*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:w*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:w*
dtype0
�
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ww*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:ww*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:w*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:w*
dtype0
�
Adam/output_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:w*'
shared_nameAdam/output_4/kernel/v
�
*Adam/output_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_4/kernel/v*
_output_shapes

:w*
dtype0
�
Adam/output_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output_4/bias/v
y
(Adam/output_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�!
value� B�  B� 
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
loss
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
 
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
�
 iter

!beta_1

"beta_2
	#decay
$learning_ratem9m:m;m<m=m>v?v@vAvBvCvD
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
�
		variables
%non_trainable_variables
&metrics
'layer_regularization_losses

(layers

trainable_variables
regularization_losses
)layer_metrics
 
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
	variables
*non_trainable_variables
+metrics

,layers
-layer_regularization_losses
regularization_losses
.layer_metrics
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
	variables
/non_trainable_variables
0metrics

1layers
2layer_regularization_losses
regularization_losses
3layer_metrics
[Y
VARIABLE_VALUEoutput_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEoutput_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
	variables
4non_trainable_variables
5metrics

6layers
7layer_regularization_losses
regularization_losses
8layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
$serving_default_input_action_matrixsPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
 serving_default_input_advantagesPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
serving_default_input_statesPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall$serving_default_input_action_matrixs serving_default_input_advantagesserving_default_input_statesdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasoutput_4/kerneloutput_4/bias*
Tin
2	*
Tout
2*+
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*/
f*R(
&__inference_signature_wrapper_54039980
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp#output_4/kernel/Read/ReadVariableOp!output_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp*Adam/output_4/kernel/m/Read/ReadVariableOp(Adam/output_4/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp*Adam/output_4/kernel/v/Read/ReadVariableOp(Adam/output_4/bias/v/Read/ReadVariableOpConst*$
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_save_54040428
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasoutput_4/kerneloutput_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/output_4/kernel/mAdam/output_4/bias/mAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/vAdam/output_4/kernel/vAdam/output_4/bias/v*#
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference__traced_restore_54040509��
�
�
E__inference_dense_6_layer_call_and_return_conditional_losses_54040235

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:w*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������w2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������w2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_my_model_2_layer_call_and_return_conditional_losses_54039870
input_states
input_action_matrixs
input_advantages
dense_6_54039854
dense_6_54039856
dense_7_54039859
dense_7_54039861
output_54039864
output_54039866
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�output/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_statesdense_6_54039854dense_6_54039856*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_540397322!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_54039859dense_7_54039861*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_540397792!
dense_7/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0output_54039864output_54039866*
Tin
2*
Tout
2*+
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_540398322 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_states:a]
+
_output_shapes
:���������
.
_user_specified_nameinput_action_matrixs:]Y
+
_output_shapes
:���������
*
_user_specified_nameinput_advantages:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
&__inference_signature_wrapper_54039980
input_action_matrixs
input_advantages
input_states
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_statesinput_action_matrixsinput_advantagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*+
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__wrapped_model_540396952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
+
_output_shapes
:���������
.
_user_specified_nameinput_action_matrixs:]Y
+
_output_shapes
:���������
*
_user_specified_nameinput_advantages:YU
+
_output_shapes
:���������
&
_user_specified_nameinput_states:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_my_model_2_layer_call_and_return_conditional_losses_54039849
input_states
input_action_matrixs
input_advantages
dense_6_54039743
dense_6_54039745
dense_7_54039790
dense_7_54039792
output_54039843
output_54039845
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�output/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_statesdense_6_54039743dense_6_54039745*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_540397322!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_54039790dense_7_54039792*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_540397792!
dense_7/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0output_54039843output_54039845*
Tin
2*
Tout
2*+
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_540398322 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_states:a]
+
_output_shapes
:���������
.
_user_specified_nameinput_action_matrixs:]Y
+
_output_shapes
:���������
*
_user_specified_nameinput_advantages:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_dense_7_layer_call_and_return_conditional_losses_54039779

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:ww*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������w2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:w*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������w2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������w2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������w:::S O
+
_output_shapes
:���������w
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_dense_6_layer_call_and_return_conditional_losses_54039732

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:w*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������w2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������w2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�p
�
H__inference_my_model_2_layer_call_and_return_conditional_losses_54040073
inputs_0
inputs_1
inputs_2-
)dense_6_tensordot_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource-
)dense_7_tensordot_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource,
(output_tensordot_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes�
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/freej
dense_6/Tensordot/ShapeShapeinputs_0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape�
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis�
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2�
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis�
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const�
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod�
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1�
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1�
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis�
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat�
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack�
dense_6/Tensordot/transpose	Transposeinputs_0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������2
dense_6/Tensordot/transpose�
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_6/Tensordot/Reshape�
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2
dense_6/Tensordot/MatMul�
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2
dense_6/Tensordot/Const_2�
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axis�
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1�
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
dense_6/Tensordot�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:w*
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2
dense_6/BiasAddt
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������w2
dense_6/Relu�
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:ww*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes�
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_7/Tensordot/free|
dense_7/Tensordot/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dense_7/Tensordot/Shape�
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis�
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2�
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis�
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const�
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod�
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1�
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1�
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis�
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat�
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack�
dense_7/Tensordot/transpose	Transposedense_6/Relu:activations:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������w2
dense_7/Tensordot/transpose�
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_7/Tensordot/Reshape�
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2
dense_7/Tensordot/MatMul�
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2
dense_7/Tensordot/Const_2�
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axis�
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1�
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
dense_7/Tensordot�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:w*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2
dense_7/BiasAddt
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������w2
dense_7/Relu�
output/Tensordot/ReadVariableOpReadVariableOp(output_tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02!
output/Tensordot/ReadVariableOpx
output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/axes
output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
output/Tensordot/freez
output/Tensordot/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
output/Tensordot/Shape�
output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/GatherV2/axis�
output/Tensordot/GatherV2GatherV2output/Tensordot/Shape:output:0output/Tensordot/free:output:0'output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2�
 output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 output/Tensordot/GatherV2_1/axis�
output/Tensordot/GatherV2_1GatherV2output/Tensordot/Shape:output:0output/Tensordot/axes:output:0)output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2_1z
output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
output/Tensordot/Const�
output/Tensordot/ProdProd"output/Tensordot/GatherV2:output:0output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
output/Tensordot/Prod~
output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
output/Tensordot/Const_1�
output/Tensordot/Prod_1Prod$output/Tensordot/GatherV2_1:output:0!output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
output/Tensordot/Prod_1~
output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
output/Tensordot/concat/axis�
output/Tensordot/concatConcatV2output/Tensordot/free:output:0output/Tensordot/axes:output:0%output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat�
output/Tensordot/stackPackoutput/Tensordot/Prod:output:0 output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/stack�
output/Tensordot/transpose	Transposedense_7/Relu:activations:0 output/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������w2
output/Tensordot/transpose�
output/Tensordot/ReshapeReshapeoutput/Tensordot/transpose:y:0output/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
output/Tensordot/Reshape�
output/Tensordot/MatMulMatMul!output/Tensordot/Reshape:output:0'output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/Tensordot/MatMul~
output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/Const_2�
output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/concat_1/axis�
output/Tensordot/concat_1ConcatV2"output/Tensordot/GatherV2:output:0!output/Tensordot/Const_2:output:0'output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat_1�
output/TensordotReshape!output/Tensordot/MatMul:product:0"output/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
output/Tensordot�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/Tensordot:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
output/BiasAdd�
output/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
output/Max/reduction_indices�

output/MaxMaxoutput/BiasAdd:output:0%output/Max/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2

output/Max�

output/subSuboutput/BiasAdd:output:0output/Max:output:0*
T0*+
_output_shapes
:���������2

output/sube

output/ExpExpoutput/sub:z:0*
T0*+
_output_shapes
:���������2

output/Exp�
output/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
output/Sum/reduction_indices�

output/SumSumoutput/Exp:y:0%output/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2

output/Sum�
output/truedivRealDivoutput/Exp:y:0output/Sum:output:0*
T0*+
_output_shapes
:���������2
output/truedivj
IdentityIdentityoutput/truediv:z:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������:::::::U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_my_model_2_layer_call_and_return_conditional_losses_54039896

inputs
inputs_1
inputs_2
dense_6_54039880
dense_6_54039882
dense_7_54039885
dense_7_54039887
output_54039890
output_54039892
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�output/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_54039880dense_6_54039882*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_540397322!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_54039885dense_7_54039887*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_540397792!
dense_7/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0output_54039890output_54039892*
Tin
2*
Tout
2*+
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_540398322 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
D__inference_output_layer_call_and_return_conditional_losses_54039832

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������w2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Max/reduction_indices�
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2
Maxg
subSubBiasAdd:output:0Max:output:0*
T0*+
_output_shapes
:���������2
subP
ExpExpsub:z:0*
T0*+
_output_shapes
:���������2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indices�
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2
Sumj
truedivRealDivExp:y:0Sum:output:0*
T0*+
_output_shapes
:���������2	
truedivc
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������w:::S O
+
_output_shapes
:���������w
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_my_model_2_layer_call_fn_54040204
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*+
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_my_model_2_layer_call_and_return_conditional_losses_540399362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_dense_7_layer_call_and_return_conditional_losses_54040275

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:ww*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������w2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:w*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������w2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������w2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������w:::S O
+
_output_shapes
:���������w
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�

*__inference_dense_7_layer_call_fn_54040284

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_540397792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������w2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������w::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������w
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_my_model_2_layer_call_fn_54040185
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*+
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_my_model_2_layer_call_and_return_conditional_losses_540398962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�?
�	
!__inference__traced_save_54040428
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop.
*savev2_output_4_kernel_read_readvariableop,
(savev2_output_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop5
1savev2_adam_output_4_kernel_m_read_readvariableop3
/savev2_adam_output_4_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop5
1savev2_adam_output_4_kernel_v_read_readvariableop3
/savev2_adam_output_4_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_7d5a42bc80904493bfc46ea7fcec70a2/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop*savev2_output_4_kernel_read_readvariableop(savev2_output_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop1savev2_adam_output_4_kernel_m_read_readvariableop/savev2_adam_output_4_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop1savev2_adam_output_4_kernel_v_read_readvariableop/savev2_adam_output_4_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :w:w:ww:w:w:: : : : : :w:w:ww:w:w::w:w:ww:w:w:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:w: 

_output_shapes
:w:$ 

_output_shapes

:ww: 

_output_shapes
:w:$ 

_output_shapes

:w: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:w: 

_output_shapes
:w:$ 

_output_shapes

:ww: 

_output_shapes
:w:$ 

_output_shapes

:w: 

_output_shapes
::$ 

_output_shapes

:w: 

_output_shapes
:w:$ 

_output_shapes

:ww: 

_output_shapes
:w:$ 

_output_shapes

:w: 

_output_shapes
::

_output_shapes
: 
�$
�
D__inference_output_layer_call_and_return_conditional_losses_54040321

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������w2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Max/reduction_indices�
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2
Maxg
subSubBiasAdd:output:0Max:output:0*
T0*+
_output_shapes
:���������2
subP
ExpExpsub:z:0*
T0*+
_output_shapes
:���������2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indices�
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2
Sumj
truedivRealDivExp:y:0Sum:output:0*
T0*+
_output_shapes
:���������2	
truedivc
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������w:::S O
+
_output_shapes
:���������w
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
~
)__inference_output_layer_call_fn_54040330

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*+
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_540398322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������w::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������w
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ۇ
�
#__inference__wrapped_model_54039695
input_states
input_action_matrixs
input_advantages8
4my_model_2_dense_6_tensordot_readvariableop_resource6
2my_model_2_dense_6_biasadd_readvariableop_resource8
4my_model_2_dense_7_tensordot_readvariableop_resource6
2my_model_2_dense_7_biasadd_readvariableop_resource7
3my_model_2_output_tensordot_readvariableop_resource5
1my_model_2_output_biasadd_readvariableop_resource
identity��
+my_model_2/dense_6/Tensordot/ReadVariableOpReadVariableOp4my_model_2_dense_6_tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02-
+my_model_2/dense_6/Tensordot/ReadVariableOp�
!my_model_2/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!my_model_2/dense_6/Tensordot/axes�
!my_model_2/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!my_model_2/dense_6/Tensordot/free�
"my_model_2/dense_6/Tensordot/ShapeShapeinput_states*
T0*
_output_shapes
:2$
"my_model_2/dense_6/Tensordot/Shape�
*my_model_2/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*my_model_2/dense_6/Tensordot/GatherV2/axis�
%my_model_2/dense_6/Tensordot/GatherV2GatherV2+my_model_2/dense_6/Tensordot/Shape:output:0*my_model_2/dense_6/Tensordot/free:output:03my_model_2/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%my_model_2/dense_6/Tensordot/GatherV2�
,my_model_2/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,my_model_2/dense_6/Tensordot/GatherV2_1/axis�
'my_model_2/dense_6/Tensordot/GatherV2_1GatherV2+my_model_2/dense_6/Tensordot/Shape:output:0*my_model_2/dense_6/Tensordot/axes:output:05my_model_2/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'my_model_2/dense_6/Tensordot/GatherV2_1�
"my_model_2/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"my_model_2/dense_6/Tensordot/Const�
!my_model_2/dense_6/Tensordot/ProdProd.my_model_2/dense_6/Tensordot/GatherV2:output:0+my_model_2/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!my_model_2/dense_6/Tensordot/Prod�
$my_model_2/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$my_model_2/dense_6/Tensordot/Const_1�
#my_model_2/dense_6/Tensordot/Prod_1Prod0my_model_2/dense_6/Tensordot/GatherV2_1:output:0-my_model_2/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#my_model_2/dense_6/Tensordot/Prod_1�
(my_model_2/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(my_model_2/dense_6/Tensordot/concat/axis�
#my_model_2/dense_6/Tensordot/concatConcatV2*my_model_2/dense_6/Tensordot/free:output:0*my_model_2/dense_6/Tensordot/axes:output:01my_model_2/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#my_model_2/dense_6/Tensordot/concat�
"my_model_2/dense_6/Tensordot/stackPack*my_model_2/dense_6/Tensordot/Prod:output:0,my_model_2/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"my_model_2/dense_6/Tensordot/stack�
&my_model_2/dense_6/Tensordot/transpose	Transposeinput_states,my_model_2/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������2(
&my_model_2/dense_6/Tensordot/transpose�
$my_model_2/dense_6/Tensordot/ReshapeReshape*my_model_2/dense_6/Tensordot/transpose:y:0+my_model_2/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2&
$my_model_2/dense_6/Tensordot/Reshape�
#my_model_2/dense_6/Tensordot/MatMulMatMul-my_model_2/dense_6/Tensordot/Reshape:output:03my_model_2/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2%
#my_model_2/dense_6/Tensordot/MatMul�
$my_model_2/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2&
$my_model_2/dense_6/Tensordot/Const_2�
*my_model_2/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*my_model_2/dense_6/Tensordot/concat_1/axis�
%my_model_2/dense_6/Tensordot/concat_1ConcatV2.my_model_2/dense_6/Tensordot/GatherV2:output:0-my_model_2/dense_6/Tensordot/Const_2:output:03my_model_2/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%my_model_2/dense_6/Tensordot/concat_1�
my_model_2/dense_6/TensordotReshape-my_model_2/dense_6/Tensordot/MatMul:product:0.my_model_2/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
my_model_2/dense_6/Tensordot�
)my_model_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp2my_model_2_dense_6_biasadd_readvariableop_resource*
_output_shapes
:w*
dtype02+
)my_model_2/dense_6/BiasAdd/ReadVariableOp�
my_model_2/dense_6/BiasAddBiasAdd%my_model_2/dense_6/Tensordot:output:01my_model_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2
my_model_2/dense_6/BiasAdd�
my_model_2/dense_6/ReluRelu#my_model_2/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������w2
my_model_2/dense_6/Relu�
+my_model_2/dense_7/Tensordot/ReadVariableOpReadVariableOp4my_model_2_dense_7_tensordot_readvariableop_resource*
_output_shapes

:ww*
dtype02-
+my_model_2/dense_7/Tensordot/ReadVariableOp�
!my_model_2/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!my_model_2/dense_7/Tensordot/axes�
!my_model_2/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!my_model_2/dense_7/Tensordot/free�
"my_model_2/dense_7/Tensordot/ShapeShape%my_model_2/dense_6/Relu:activations:0*
T0*
_output_shapes
:2$
"my_model_2/dense_7/Tensordot/Shape�
*my_model_2/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*my_model_2/dense_7/Tensordot/GatherV2/axis�
%my_model_2/dense_7/Tensordot/GatherV2GatherV2+my_model_2/dense_7/Tensordot/Shape:output:0*my_model_2/dense_7/Tensordot/free:output:03my_model_2/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%my_model_2/dense_7/Tensordot/GatherV2�
,my_model_2/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,my_model_2/dense_7/Tensordot/GatherV2_1/axis�
'my_model_2/dense_7/Tensordot/GatherV2_1GatherV2+my_model_2/dense_7/Tensordot/Shape:output:0*my_model_2/dense_7/Tensordot/axes:output:05my_model_2/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'my_model_2/dense_7/Tensordot/GatherV2_1�
"my_model_2/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"my_model_2/dense_7/Tensordot/Const�
!my_model_2/dense_7/Tensordot/ProdProd.my_model_2/dense_7/Tensordot/GatherV2:output:0+my_model_2/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!my_model_2/dense_7/Tensordot/Prod�
$my_model_2/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$my_model_2/dense_7/Tensordot/Const_1�
#my_model_2/dense_7/Tensordot/Prod_1Prod0my_model_2/dense_7/Tensordot/GatherV2_1:output:0-my_model_2/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#my_model_2/dense_7/Tensordot/Prod_1�
(my_model_2/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(my_model_2/dense_7/Tensordot/concat/axis�
#my_model_2/dense_7/Tensordot/concatConcatV2*my_model_2/dense_7/Tensordot/free:output:0*my_model_2/dense_7/Tensordot/axes:output:01my_model_2/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#my_model_2/dense_7/Tensordot/concat�
"my_model_2/dense_7/Tensordot/stackPack*my_model_2/dense_7/Tensordot/Prod:output:0,my_model_2/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"my_model_2/dense_7/Tensordot/stack�
&my_model_2/dense_7/Tensordot/transpose	Transpose%my_model_2/dense_6/Relu:activations:0,my_model_2/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������w2(
&my_model_2/dense_7/Tensordot/transpose�
$my_model_2/dense_7/Tensordot/ReshapeReshape*my_model_2/dense_7/Tensordot/transpose:y:0+my_model_2/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2&
$my_model_2/dense_7/Tensordot/Reshape�
#my_model_2/dense_7/Tensordot/MatMulMatMul-my_model_2/dense_7/Tensordot/Reshape:output:03my_model_2/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2%
#my_model_2/dense_7/Tensordot/MatMul�
$my_model_2/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2&
$my_model_2/dense_7/Tensordot/Const_2�
*my_model_2/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*my_model_2/dense_7/Tensordot/concat_1/axis�
%my_model_2/dense_7/Tensordot/concat_1ConcatV2.my_model_2/dense_7/Tensordot/GatherV2:output:0-my_model_2/dense_7/Tensordot/Const_2:output:03my_model_2/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%my_model_2/dense_7/Tensordot/concat_1�
my_model_2/dense_7/TensordotReshape-my_model_2/dense_7/Tensordot/MatMul:product:0.my_model_2/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
my_model_2/dense_7/Tensordot�
)my_model_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp2my_model_2_dense_7_biasadd_readvariableop_resource*
_output_shapes
:w*
dtype02+
)my_model_2/dense_7/BiasAdd/ReadVariableOp�
my_model_2/dense_7/BiasAddBiasAdd%my_model_2/dense_7/Tensordot:output:01my_model_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2
my_model_2/dense_7/BiasAdd�
my_model_2/dense_7/ReluRelu#my_model_2/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������w2
my_model_2/dense_7/Relu�
*my_model_2/output/Tensordot/ReadVariableOpReadVariableOp3my_model_2_output_tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02,
*my_model_2/output/Tensordot/ReadVariableOp�
 my_model_2/output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 my_model_2/output/Tensordot/axes�
 my_model_2/output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 my_model_2/output/Tensordot/free�
!my_model_2/output/Tensordot/ShapeShape%my_model_2/dense_7/Relu:activations:0*
T0*
_output_shapes
:2#
!my_model_2/output/Tensordot/Shape�
)my_model_2/output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)my_model_2/output/Tensordot/GatherV2/axis�
$my_model_2/output/Tensordot/GatherV2GatherV2*my_model_2/output/Tensordot/Shape:output:0)my_model_2/output/Tensordot/free:output:02my_model_2/output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$my_model_2/output/Tensordot/GatherV2�
+my_model_2/output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+my_model_2/output/Tensordot/GatherV2_1/axis�
&my_model_2/output/Tensordot/GatherV2_1GatherV2*my_model_2/output/Tensordot/Shape:output:0)my_model_2/output/Tensordot/axes:output:04my_model_2/output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&my_model_2/output/Tensordot/GatherV2_1�
!my_model_2/output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!my_model_2/output/Tensordot/Const�
 my_model_2/output/Tensordot/ProdProd-my_model_2/output/Tensordot/GatherV2:output:0*my_model_2/output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 my_model_2/output/Tensordot/Prod�
#my_model_2/output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#my_model_2/output/Tensordot/Const_1�
"my_model_2/output/Tensordot/Prod_1Prod/my_model_2/output/Tensordot/GatherV2_1:output:0,my_model_2/output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"my_model_2/output/Tensordot/Prod_1�
'my_model_2/output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'my_model_2/output/Tensordot/concat/axis�
"my_model_2/output/Tensordot/concatConcatV2)my_model_2/output/Tensordot/free:output:0)my_model_2/output/Tensordot/axes:output:00my_model_2/output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"my_model_2/output/Tensordot/concat�
!my_model_2/output/Tensordot/stackPack)my_model_2/output/Tensordot/Prod:output:0+my_model_2/output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!my_model_2/output/Tensordot/stack�
%my_model_2/output/Tensordot/transpose	Transpose%my_model_2/dense_7/Relu:activations:0+my_model_2/output/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������w2'
%my_model_2/output/Tensordot/transpose�
#my_model_2/output/Tensordot/ReshapeReshape)my_model_2/output/Tensordot/transpose:y:0*my_model_2/output/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2%
#my_model_2/output/Tensordot/Reshape�
"my_model_2/output/Tensordot/MatMulMatMul,my_model_2/output/Tensordot/Reshape:output:02my_model_2/output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"my_model_2/output/Tensordot/MatMul�
#my_model_2/output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#my_model_2/output/Tensordot/Const_2�
)my_model_2/output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)my_model_2/output/Tensordot/concat_1/axis�
$my_model_2/output/Tensordot/concat_1ConcatV2-my_model_2/output/Tensordot/GatherV2:output:0,my_model_2/output/Tensordot/Const_2:output:02my_model_2/output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$my_model_2/output/Tensordot/concat_1�
my_model_2/output/TensordotReshape,my_model_2/output/Tensordot/MatMul:product:0-my_model_2/output/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
my_model_2/output/Tensordot�
(my_model_2/output/BiasAdd/ReadVariableOpReadVariableOp1my_model_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(my_model_2/output/BiasAdd/ReadVariableOp�
my_model_2/output/BiasAddBiasAdd$my_model_2/output/Tensordot:output:00my_model_2/output/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
my_model_2/output/BiasAdd�
'my_model_2/output/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'my_model_2/output/Max/reduction_indices�
my_model_2/output/MaxMax"my_model_2/output/BiasAdd:output:00my_model_2/output/Max/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2
my_model_2/output/Max�
my_model_2/output/subSub"my_model_2/output/BiasAdd:output:0my_model_2/output/Max:output:0*
T0*+
_output_shapes
:���������2
my_model_2/output/sub�
my_model_2/output/ExpExpmy_model_2/output/sub:z:0*
T0*+
_output_shapes
:���������2
my_model_2/output/Exp�
'my_model_2/output/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'my_model_2/output/Sum/reduction_indices�
my_model_2/output/SumSummy_model_2/output/Exp:y:00my_model_2/output/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2
my_model_2/output/Sum�
my_model_2/output/truedivRealDivmy_model_2/output/Exp:y:0my_model_2/output/Sum:output:0*
T0*+
_output_shapes
:���������2
my_model_2/output/truedivu
IdentityIdentitymy_model_2/output/truediv:z:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������:::::::Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_states:a]
+
_output_shapes
:���������
.
_user_specified_nameinput_action_matrixs:]Y
+
_output_shapes
:���������
*
_user_specified_nameinput_advantages:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_my_model_2_layer_call_and_return_conditional_losses_54039936

inputs
inputs_1
inputs_2
dense_6_54039920
dense_6_54039922
dense_7_54039925
dense_7_54039927
output_54039930
output_54039932
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�output/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_54039920dense_6_54039922*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_540397322!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_54039925dense_7_54039927*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_540397792!
dense_7/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0output_54039930output_54039932*
Tin
2*
Tout
2*+
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_540398322 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_my_model_2_layer_call_fn_54039951
input_states
input_action_matrixs
input_advantages
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_statesinput_action_matrixsinput_advantagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*+
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_my_model_2_layer_call_and_return_conditional_losses_540399362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_states:a]
+
_output_shapes
:���������
.
_user_specified_nameinput_action_matrixs:]Y
+
_output_shapes
:���������
*
_user_specified_nameinput_advantages:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�h
�
$__inference__traced_restore_54040509
file_prefix#
assignvariableop_dense_6_kernel#
assignvariableop_1_dense_6_bias%
!assignvariableop_2_dense_7_kernel#
assignvariableop_3_dense_7_bias&
"assignvariableop_4_output_4_kernel$
 assignvariableop_5_output_4_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate-
)assignvariableop_11_adam_dense_6_kernel_m+
'assignvariableop_12_adam_dense_6_bias_m-
)assignvariableop_13_adam_dense_7_kernel_m+
'assignvariableop_14_adam_dense_7_bias_m.
*assignvariableop_15_adam_output_4_kernel_m,
(assignvariableop_16_adam_output_4_bias_m-
)assignvariableop_17_adam_dense_6_kernel_v+
'assignvariableop_18_adam_dense_6_bias_v-
)assignvariableop_19_adam_dense_7_kernel_v+
'assignvariableop_20_adam_dense_7_bias_v.
*assignvariableop_21_adam_output_4_kernel_v,
(assignvariableop_22_adam_output_4_bias_v
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_7_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_7_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_output_4_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_output_4_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_6_kernel_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_dense_6_bias_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_dense_7_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_7_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_output_4_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_output_4_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_6_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_6_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_7_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_7_bias_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_output_4_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_output_4_bias_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23�
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_my_model_2_layer_call_fn_54039911
input_states
input_action_matrixs
input_advantages
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_statesinput_action_matrixsinput_advantagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*+
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_my_model_2_layer_call_and_return_conditional_losses_540398962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_states:a]
+
_output_shapes
:���������
.
_user_specified_nameinput_action_matrixs:]Y
+
_output_shapes
:���������
*
_user_specified_nameinput_advantages:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�p
�
H__inference_my_model_2_layer_call_and_return_conditional_losses_54040166
inputs_0
inputs_1
inputs_2-
)dense_6_tensordot_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource-
)dense_7_tensordot_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource,
(output_tensordot_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes�
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/freej
dense_6/Tensordot/ShapeShapeinputs_0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape�
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis�
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2�
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis�
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const�
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod�
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1�
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1�
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis�
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat�
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack�
dense_6/Tensordot/transpose	Transposeinputs_0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������2
dense_6/Tensordot/transpose�
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_6/Tensordot/Reshape�
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2
dense_6/Tensordot/MatMul�
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2
dense_6/Tensordot/Const_2�
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axis�
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1�
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
dense_6/Tensordot�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:w*
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2
dense_6/BiasAddt
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������w2
dense_6/Relu�
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:ww*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes�
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_7/Tensordot/free|
dense_7/Tensordot/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dense_7/Tensordot/Shape�
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis�
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2�
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis�
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const�
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod�
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1�
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1�
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis�
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat�
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack�
dense_7/Tensordot/transpose	Transposedense_6/Relu:activations:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������w2
dense_7/Tensordot/transpose�
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_7/Tensordot/Reshape�
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w2
dense_7/Tensordot/MatMul�
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w2
dense_7/Tensordot/Const_2�
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axis�
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1�
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������w2
dense_7/Tensordot�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:w*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������w2
dense_7/BiasAddt
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������w2
dense_7/Relu�
output/Tensordot/ReadVariableOpReadVariableOp(output_tensordot_readvariableop_resource*
_output_shapes

:w*
dtype02!
output/Tensordot/ReadVariableOpx
output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/axes
output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
output/Tensordot/freez
output/Tensordot/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
output/Tensordot/Shape�
output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/GatherV2/axis�
output/Tensordot/GatherV2GatherV2output/Tensordot/Shape:output:0output/Tensordot/free:output:0'output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2�
 output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 output/Tensordot/GatherV2_1/axis�
output/Tensordot/GatherV2_1GatherV2output/Tensordot/Shape:output:0output/Tensordot/axes:output:0)output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2_1z
output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
output/Tensordot/Const�
output/Tensordot/ProdProd"output/Tensordot/GatherV2:output:0output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
output/Tensordot/Prod~
output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
output/Tensordot/Const_1�
output/Tensordot/Prod_1Prod$output/Tensordot/GatherV2_1:output:0!output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
output/Tensordot/Prod_1~
output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
output/Tensordot/concat/axis�
output/Tensordot/concatConcatV2output/Tensordot/free:output:0output/Tensordot/axes:output:0%output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat�
output/Tensordot/stackPackoutput/Tensordot/Prod:output:0 output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/stack�
output/Tensordot/transpose	Transposedense_7/Relu:activations:0 output/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������w2
output/Tensordot/transpose�
output/Tensordot/ReshapeReshapeoutput/Tensordot/transpose:y:0output/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
output/Tensordot/Reshape�
output/Tensordot/MatMulMatMul!output/Tensordot/Reshape:output:0'output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/Tensordot/MatMul~
output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/Const_2�
output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/concat_1/axis�
output/Tensordot/concat_1ConcatV2"output/Tensordot/GatherV2:output:0!output/Tensordot/Const_2:output:0'output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat_1�
output/TensordotReshape!output/Tensordot/MatMul:product:0"output/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
output/Tensordot�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/Tensordot:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
output/BiasAdd�
output/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
output/Max/reduction_indices�

output/MaxMaxoutput/BiasAdd:output:0%output/Max/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2

output/Max�

output/subSuboutput/BiasAdd:output:0output/Max:output:0*
T0*+
_output_shapes
:���������2

output/sube

output/ExpExpoutput/sub:z:0*
T0*+
_output_shapes
:���������2

output/Exp�
output/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
output/Sum/reduction_indices�

output/SumSumoutput/Exp:y:0%output/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(2

output/Sum�
output/truedivRealDivoutput/Exp:y:0output/Sum:output:0*
T0*+
_output_shapes
:���������2
output/truedivj
IdentityIdentityoutput/truediv:z:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:���������:���������:���������:::::::U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

*__inference_dense_6_layer_call_fn_54040244

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*+
_output_shapes
:���������w*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_540397322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������w2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Y
input_action_matrixsA
&serving_default_input_action_matrixs:0���������
Q
input_advantages=
"serving_default_input_advantages:0���������
I
input_states9
serving_default_input_states:0���������>
output4
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�.
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
loss
		variables

trainable_variables
regularization_losses
	keras_api

signatures
*E&call_and_return_all_conditional_losses
F_default_save_signature
G__call__"�+
_tf_keras_model�+{"class_name": "MyModel", "name": "my_model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "my_model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_states"}, "name": "input_states", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 119, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["input_states", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 119, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_action_matrixs"}, "name": "input_action_matrixs", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_advantages"}, "name": "input_advantages", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}], "input_layers": [["input_states", 0, 0], ["input_action_matrixs", 0, 0], ["input_advantages", 0, 0]], "output_layers": [["output", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 4]}, {"class_name": "TensorShape", "items": [null, 1, 2]}, {"class_name": "TensorShape", "items": [null, 1, 1]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "MyModel", "config": {"name": "my_model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_states"}, "name": "input_states", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 119, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["input_states", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 119, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_action_matrixs"}, "name": "input_action_matrixs", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_advantages"}, "name": "input_advantages", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}], "input_layers": [["input_states", 0, 0], ["input_action_matrixs", 0, 0], ["input_advantages", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0007044665399007499, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_states", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_states"}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 119, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 4]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 119, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 119}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 119]}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_action_matrixs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_action_matrixs"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_advantages", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_advantages"}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 119}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 119]}}
�
 iter

!beta_1

"beta_2
	#decay
$learning_ratem9m:m;m<m=m>v?v@vAvBvCvD"
	optimizer
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
		variables
%non_trainable_variables
&metrics
'layer_regularization_losses

(layers

trainable_variables
regularization_losses
)layer_metrics
G__call__
F_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Nserving_default"
signature_map
 :w2dense_6/kernel
:w2dense_6/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
*non_trainable_variables
+metrics

,layers
-layer_regularization_losses
regularization_losses
.layer_metrics
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 :ww2dense_7/kernel
:w2dense_7/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
/non_trainable_variables
0metrics

1layers
2layer_regularization_losses
regularization_losses
3layer_metrics
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
!:w2output_4/kernel
:2output_4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
4non_trainable_variables
5metrics

6layers
7layer_regularization_losses
regularization_losses
8layer_metrics
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
%:#w2Adam/dense_6/kernel/m
:w2Adam/dense_6/bias/m
%:#ww2Adam/dense_7/kernel/m
:w2Adam/dense_7/bias/m
&:$w2Adam/output_4/kernel/m
 :2Adam/output_4/bias/m
%:#w2Adam/dense_6/kernel/v
:w2Adam/dense_6/bias/v
%:#ww2Adam/dense_7/kernel/v
:w2Adam/dense_7/bias/v
&:$w2Adam/output_4/kernel/v
 :2Adam/output_4/bias/v
�2�
H__inference_my_model_2_layer_call_and_return_conditional_losses_54039849
H__inference_my_model_2_layer_call_and_return_conditional_losses_54039870
H__inference_my_model_2_layer_call_and_return_conditional_losses_54040073
H__inference_my_model_2_layer_call_and_return_conditional_losses_54040166�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference__wrapped_model_54039695�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���
*�'
input_states���������
2�/
input_action_matrixs���������
.�+
input_advantages���������
�2�
-__inference_my_model_2_layer_call_fn_54039911
-__inference_my_model_2_layer_call_fn_54039951
-__inference_my_model_2_layer_call_fn_54040204
-__inference_my_model_2_layer_call_fn_54040185�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_dense_6_layer_call_and_return_conditional_losses_54040235�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_6_layer_call_fn_54040244�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_7_layer_call_and_return_conditional_losses_54040275�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_7_layer_call_fn_54040284�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_output_layer_call_and_return_conditional_losses_54040321�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_output_layer_call_fn_54040330�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
`B^
&__inference_signature_wrapper_54039980input_action_matrixsinput_advantagesinput_states�
#__inference__wrapped_model_54039695����
���
���
*�'
input_states���������
2�/
input_action_matrixs���������
.�+
input_advantages���������
� "3�0
.
output$�!
output����������
E__inference_dense_6_layer_call_and_return_conditional_losses_54040235d3�0
)�&
$�!
inputs���������
� ")�&
�
0���������w
� �
*__inference_dense_6_layer_call_fn_54040244W3�0
)�&
$�!
inputs���������
� "����������w�
E__inference_dense_7_layer_call_and_return_conditional_losses_54040275d3�0
)�&
$�!
inputs���������w
� ")�&
�
0���������w
� �
*__inference_dense_7_layer_call_fn_54040284W3�0
)�&
$�!
inputs���������w
� "����������w�
H__inference_my_model_2_layer_call_and_return_conditional_losses_54039849����
���
���
*�'
input_states���������
2�/
input_action_matrixs���������
.�+
input_advantages���������
p

 
� ")�&
�
0���������
� �
H__inference_my_model_2_layer_call_and_return_conditional_losses_54039870����
���
���
*�'
input_states���������
2�/
input_action_matrixs���������
.�+
input_advantages���������
p 

 
� ")�&
�
0���������
� �
H__inference_my_model_2_layer_call_and_return_conditional_losses_54040073����
���
{�x
&�#
inputs/0���������
&�#
inputs/1���������
&�#
inputs/2���������
p

 
� ")�&
�
0���������
� �
H__inference_my_model_2_layer_call_and_return_conditional_losses_54040166����
���
{�x
&�#
inputs/0���������
&�#
inputs/1���������
&�#
inputs/2���������
p 

 
� ")�&
�
0���������
� �
-__inference_my_model_2_layer_call_fn_54039911����
���
���
*�'
input_states���������
2�/
input_action_matrixs���������
.�+
input_advantages���������
p

 
� "�����������
-__inference_my_model_2_layer_call_fn_54039951����
���
���
*�'
input_states���������
2�/
input_action_matrixs���������
.�+
input_advantages���������
p 

 
� "�����������
-__inference_my_model_2_layer_call_fn_54040185����
���
{�x
&�#
inputs/0���������
&�#
inputs/1���������
&�#
inputs/2���������
p

 
� "�����������
-__inference_my_model_2_layer_call_fn_54040204����
���
{�x
&�#
inputs/0���������
&�#
inputs/1���������
&�#
inputs/2���������
p 

 
� "�����������
D__inference_output_layer_call_and_return_conditional_losses_54040321d3�0
)�&
$�!
inputs���������w
� ")�&
�
0���������
� �
)__inference_output_layer_call_fn_54040330W3�0
)�&
$�!
inputs���������w
� "�����������
&__inference_signature_wrapper_54039980����
� 
���
J
input_action_matrixs2�/
input_action_matrixs���������
B
input_advantages.�+
input_advantages���������
:
input_states*�'
input_states���������"3�0
.
output$�!
output���������