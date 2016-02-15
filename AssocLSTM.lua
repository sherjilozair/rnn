local AssocLSTM, parent = torch.class("nn.AssocLSTM", "nn.LSTM")


function AssocLSTM:__init(inputSize, outputSize, rho)
   parent.__init(self, inputSize, outputSize, rho, false)
end

function AssocLSTM:buildModel()
   -- input : {input, prevOutput, prevCell}
   -- output : {output, cell}
   
   require 'nngraph'
   
   -- Calculate all four gates in one go : input, hidden, forget, output
   self.i2g = nn.Linear(self.inputSize, 5*self.outputSize)
   self.o2g = nn.LinearNoBias(self.outputSize, 5*self.outputSize)
   
   return self:nngraphModel()
   
end

function AssocLSTM:nngraphModel()
   assert(nngraph, "Missing nngraph package")
   
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   table.insert(inputs, nn.Identity()()) -- prev_h[L]
   table.insert(inputs, nn.Identity()()) -- prev_c[L]
   
   local x, prev_h, prev_c = unpack(inputs)
   
   -- evaluate the input sums at once for efficiency
   local i2h = self.i2g(x):annotate{name='i2h'}
   local h2h = self.o2g(prev_h):annotate{name='h2h'}
   local all_input_sums = nn.CAddTable()({i2h, h2h})

   local reshaped = nn.Reshape(5, self.outputSize)(all_input_sums)
   -- input, hidden, forget, output
   local n1, n2, n3, n4, n5 = nn.SplitTable(2)(reshaped):split(5)
   local in_gate = nn.Sigmoid()(n1)
   local forget_gate = nn.Sigmoid()(n2)
   local out_gate = nn.Sigmoid()(n3)
   local in_key_gate = nn.Sigmoid()(n4)
   local out_key_gate = nn.Sigmoid()(n5)

   -- Assoc-LSTM computation needed here
   -- perform the LSTM update
   local next_c           = nn.CAddTable()({
     nn.CMulTable()({forget_gate, prev_c}),
     nn.CMulTable()({in_gate,     in_transform})
   })
   -- gated cells form the output
   local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
   -- till here

   local outputs = {next_h, next_c}
   
   return nn.gModule(inputs, outputs)
end

function AssocLSTM:buildGate()
   error"Not Implemented"
end

function AssocLSTM:buildInputGate()
   error"Not Implemented"
end

function AssocLSTM:buildForgetGate()
   error"Not Implemented"
end

function AssocLSTM:buildHidden()
   error"Not Implemented"
end

function AssocLSTM:buildCell()
   error"Not Implemented"
end   
   
function AssocLSTM:buildOutputGate()
   error"Not Implemented"
end
