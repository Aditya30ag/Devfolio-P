const mongoose=require('mongoose');
const {Schema} =mongoose;

const Adminschema=new Schema({
    name:{
        type:String,
        required:true
    },
    email:{
        type:String,
        required:true,
    },
    password:{
        type:String,
        required:true
    }
})
const admin=mongoose.model("Admin",Adminschema);
module.exports=admin