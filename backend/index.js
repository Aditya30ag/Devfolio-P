const connectToMongo=require('./db');
const express = require('express');
var cors = require('cors');
require('dotenv').config();

connectToMongo();
const app = express()
const port = process.env.PORT || 5000;

app.use(express.json());
app.use(cors());

app.use('/api/auth', require('./routes/auth'))
app.use('/api/notes', require('./routes/notes'))
app.use('/api/admin', require('./routes/admin'))

app.listen(port, () => {
  console.log(`Example app listening on port http://localhost:${port}`)
})