const mongoose=require('mongoose');
const {Schema} =mongoose;

const Userschema=new Schema({
    aadharNumber:{
        type:Number,
        required:true,
    },
    voterid:{
        type:Number,
        required:true,
    },
    isvoted:{
        type:Boolean,
        required:false,
        default:false
    },
})
const user=mongoose.model("User",Userschema);
module.exports=user